# sahi4detrex
添加部分内容，使得sahi支持detrex框架下的检测器进行推理

1. 需要以可修改的形式( pip -e . )安装detrex和sahi
2. 为原始sahi项目中的sahi/auto_model.py文件 和sahi/models/__init__.py文件添加detrex类
3. 在sahi/models、中添加detrex.py文件，定义DetrexDetectionModel 类
4. 在demo中添加了inference_for_detrex.ipynb进行验证。

为了能够使用detrex中的配置文件更好的实例化DetrexDetectionModel 类，我们使用detectron2.config.instantiate和detrex.demo.predictors.DefaultPredictor
因此需要将detrex/demo/拷贝到detrex/detrex中。