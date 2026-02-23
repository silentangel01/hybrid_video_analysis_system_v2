// server.js
import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// 获取当前目录
const __dirname = dirname(fileURLToPath(import.meta.url));

// 创建 Express 应用
const app = express();

// 添加 CORS 中间件
/**
 * 添加了Access-Control-Allow-Origin头
 * 允许POST和OPTIONS方法
 * 处理预检请求（OPTIONS）
 */
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'http://localhost:5173');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.header('Access-Control-Allow-Credentials', 'true');

    // 处理预检请求（OPTIONS）
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    next();
});

// -------------------------- 文件上传 --------------------------
// 配置文件存储路径
// 临时保存到 ./uploads/
const upload = multer({ dest: './uploads' });

// 定义上传路由
app.post('/upload', upload.single('video'), (req, res) => {
    // 获取原始文件名
    const { originalname } = req.file;
    // 用户选择的类型：parking / smoke_flame / common_space
    const folder = req.body.type;

    // 构造目标路径
    const targetPath = path.join(__dirname, '../backend/uploads', folder, originalname);

    // 确保目标目录存在
    const targetDir = path.dirname(targetPath);
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
    }

    // 移动文件到正确目录
    // req.file.path是临时文件路径（由multer生成）
    // targetPath是最终要保存的位置（指向backend/uploads/xxx/）
    // 使用fs.rename()移动文件（比复制更高效）
    fs.rename(req.file.path, targetPath, (err) => {
        if (err) {
            console.error('Error moving file:', err);
            return res.status(500).json({ error: 'Failed to save video' });
        }
        console.log(`✅ Video saved to: ${targetPath}`);
        res.json({ success: true, message: 'Video uploaded successfully!' });
    });
});

// ================== 新增：查询火情事件 ==================
import { MongoClient } from 'mongodb';

const MONGO_URI = 'mongodb://localhost:27017'; // 宿主机访问
const DB_NAME = 'video_analysis_db';

app.get('/api/events', async (req, res) => {
    try {
        const client = new MongoClient(MONGO_URI);
        await client.connect();
        const db = client.db(DB_NAME);
        const collection = db.collection('events'); // 确保集合名正确

        // 查询最近5分钟内、类型为 smoke_flame 的事件
        // 使用浮点数比较：将当前时间转换为浮点数
        const now = Date.now() / 1000; // 转换为秒级浮点数
        const fiveMinutesAgo = now - 300; // 5分钟前（单位：秒）

        const events = await collection
            .find({
                event_type: { $in: ['fire', 'smoke'] }, // 匹配 fire 或 smoke
                timestamp: { $gte: fiveMinutesAgo }
            })
            .sort({ timestamp: -1 })
            .limit(10)
            .toArray();

        console.log('🔍 查询到的事件:', events); // 添加调试日志

        // 提取唯一的 cameraId（即视频文件名）
        // 过滤掉 null/undefined 的 camera_id
        const uniqueSources = [...new Set(events.filter(e => e.camera_id).map(e => e.camera_id))];

        await client.close();

        res.json({
            success: true,
            fireDetected: uniqueSources.length > 0,
            sources: uniqueSources
        });
    } catch (error) {
        console.error('MongoDB query error:', error);
        res.status(500).json({ success: false, error: 'Database query failed' });
    }
});

// ================== 新增：查询所有事件（用于 EventList） ==================
app.get('/api/events-all', async (req, res) => {
    try {
        const client = new MongoClient(MONGO_URI);
        await client.connect();
        const db = client.db(DB_NAME);
        const collection = db.collection('events');

        const events = await collection
            .find({})
            .sort({ timestamp: -1 })
            .limit(50)
            .toArray();

        // 转换 ObjectId 为字符串（避免 JSON 序列化问题）
        const safeEvents = events.map(event => ({
            ...event,
            _id: event._id.toString(),
            timestamp: event.timestamp // 浮点数，前端可处理
        }));

        await client.close();

        res.json({
            success: true,
            events: safeEvents
        });
    } catch (error) {
        console.error('Fetch all events error:', error);
        res.status(500).json({ success: false, error: 'Failed to fetch events' });
    }
});

// ================== 新增：代理 MinIO 图片请求 ==================
import axios from 'axios';

app.get('/api/image/:key', async (req, res) => {
    const { key } = req.params;
    const imageUrl = `http://localhost:9000/${key}`; // MinIO 地址

    try {
        // 发起请求到 MinIO
        const imageResponse = await axios({
            method: 'GET',
            url: imageUrl,
            responseType: 'stream',
            headers: {
                'Authorization': 'Bearer your-minio-token' // 如果需要认证，否则删除这行
            }
        });

        // 设置响应头（如 content-type）
        res.set('Content-Type', imageResponse.headers['content-type'] || 'image/jpeg');

        // 流式传输图片
        imageResponse.data.pipe(res);
    } catch (error) {
        console.error('图片获取失败:', error.response?.data || error.message);
        res.status(404).send('图片未找到');
    }
});

// -------------------------- 启动服务器 --------------------------
const PORT = 8080;
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});