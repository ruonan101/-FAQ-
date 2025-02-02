# FAQ智能客服系统

这是一个基于嵌入(Embedding)技术的智能客服系统，支持：
- FAQ的添加和管理
- 智能问答匹配
- 当没有合适答案时转人工服务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python main.py
```

服务启动后，访问 http://localhost:8000 即可使用系统。

## 系统功能

1. FAQ管理：
   - 添加新的问答对
   - 查看现有FAQ列表

2. 智能问答：
   - 输入问题获取匹配答案
   - 相似度低于阈值时会建议转人工服务

## 技术说明

- 使用sentence-transformers进行文本嵌入
- 使用ChromaDB作为向量数据库
- 基于FastAPI构建后端API
- 使用Bootstrap构建前端界面

## 许可证
本项目采用 MIT 许可证。查看 [LICENSE](LICENSE) 文件了解更多详细信息。

主要权限：
- ✅ 商业用途
- ✅ 修改
- ✅ 分发
- ✅ 私人使用

主要限制：
- ❗ 责任限制
- ❗ 不提供担保

主要条件：
- ℹ️ 包含许可证声明
- ℹ️ 包含版权声明
