# 直接编辑、右键运行

本目录包含三个独立的 Python 测试入口和已经准备好的默认输入。无需启动服务、配置启动参数或修改脚本变量。

| 右键 Run / Debug 的文件 | 直接修改的问题与候选 | 可直接替换的素材 |
| --- | --- | --- |
| `test_text.py` | `text.txt` | 文本已在 `text.txt` 中 |
| `test_image.py` | `image.txt` | `image.jpg` |
| `test_video.py` | `video.txt` | `video.mp4`，m3bench 的 30 秒片段 |

PyCharm 解释器选项目根目录下 `.venv/bin/python`，打开相应 `.py`，右键 Run / Debug 即可。
在脚本的 `return result` 打断点可查看完整结果；Step Into 可进入加载、预处理和模型前向。
脚本固定读取同目录下的文件，不受 IDE 工作目录影响。

修改 `.txt` 时，前面写上下文和问题，末尾每行写一个候选，使用连续的 `A. `、`B. `、`C. ` 等标签（至少两项）。
替换图片或视频时保持文件名不变，再运行对应脚本。图片如换成其他格式，可以先另存为 JPEG。
视频只分析画面，不识别音轨；默认固定采样 8 帧，所以短暂发生的动作可能漏采。

默认视频取自已有 m3bench `bedroom_01` 派生片段 `clip002.mp4` 的前 30 秒，
与此前自测的 kitchen/living 片段不同。已转为 H.264 MP4、30fps、900 帧、30.000 秒，去除音轨。
来源、转换参数和 SHA256 记录在 `provenance.json`。训练数据清单不在本机，无法确认该视频是否
参与过 Raya 训练；它可用于自定义测试，但不能据此称为严格的未见数据泛化评测。

图片和视频已在本机准备好，但不提交到 Git。换机器需要拷贝这两个文件；若原 bench 仍在，
也可按 `provenance.json` 的路径和转换参数重建。原有自动化 HTTP 测试素材仍保留在 `tests/fixtures/`。
