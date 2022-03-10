using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using System.IO;
using OpenCvSharp.Dnn;

namespace NetYoloV3
{
    public class YoloV3
    {
        private readonly YoloV3Config _config;

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="pathModel"> 权重路径</param>
        /// <param name="pathConfig">权重参数路径</param>
        /// <param name="labelsFile">标签路径</param>
        /// <param name="imgWidth">yolo模型要求的的图片大小</param>
        /// <param name="imgHigh">yolo模型要求的的图片大小</param>
        /// <param name="threshold">置信度阈值</param>
        /// <param name="nms">nms 阈值</param>
        public YoloV3(string pathModel, string pathConfig, string labelsFile,
                       int imgWidth = 320, int imgHigh = 320, float threshold = 0.5f, float nms = 0.5f)
        {
            _config = new YoloV3Config
            {
                ModelWeights        = pathModel,
                ModelConfiguaration = pathConfig,
                LabelsFile          = labelsFile,
                ImgWidth            = imgWidth,
                ImgHight            = imgHigh,
                Threshold           = threshold,
                NmsThreshold        = nms
            };
        }

        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="imgpath">图像路径</param>
        /// <returns></returns>
        public NetResult[] Detect(string imgpath)
        {
            Mat img = Cv2.ImRead(imgpath);
            return Process(img);
        }

        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        public NetResult[] Detect(System.Drawing.Bitmap img)
        {
            Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img);
            return Process(mat);
        }


        #region way
        /// <summary>
        /// 图片预处理
        /// </summary>
        /// <param Mat img</param>
        /// <returns></returns>
        private Mat ImagePretreatment(Mat img)
        {
            double scale = 1.0 / 255;                     //像素大小由0~255范围变为0~1，深度学习中的输入都是0~1范围的，便于优化，暂时每一其他数值大小的       
            Size size = new Size(_config.ImgWidth, _config.ImgHight);

            //yolo的网络结果是使用同大小一图像进行训练的所以需要将图像转为对应的大小
            //生成blob, 块尺寸可以是320/416/608
            Mat blob = CvDnn.BlobFromImage(img, scale, size, new Scalar(), true, false);

            return blob;
        }

        /// <summary>
        /// 读入标签
        /// </summary>
        /// <param name="pathLabels"></param>
        /// <returns></returns>
        private string[] ReadLabels(string pathLabels)
        {
            if (!File.Exists(pathLabels))
                throw new FileNotFoundException("The file of labels not foud", pathLabels);

            string[] classNames = File.ReadAllLines(pathLabels).ToArray();

            return classNames;
        }

        /// <summary>
        /// 初始化模型
        /// </summary>
        private Net InitializeModel(string pathModel, string pathConfig)
        {
            if (!File.Exists(pathModel))
                throw new FileNotFoundException("The file model has not found", pathModel);
            if (!File.Exists(pathConfig))
                throw new FileNotFoundException("The file config has not found", pathConfig);

            Net net = CvDnn.ReadNetFromDarknet(pathConfig, pathModel);
            if (net == null || net.Empty())
                throw new Exception("The model has not yet initialized or is empty.");

            //读入模型和设置
            net.SetPreferableBackend(Backend.OPENCV);     // 3:DNN_BACKEND_OPENCV 
            net.SetPreferableTarget(Target.CPU);          //dnn target cpu

            return net;
        }

        /// <summary>
        /// 检测的后处理
        /// </summary>
        /// <param name="image"></param>
        /// <param name="results"></param>
        /// <returns></returns>
        private NetResult[] Postprocess(ref Mat image, Mat[] results)
        {
            var netResults = new List<NetResult>();         //存储检测结果

            var classIds = new List<int>();                 //可能对象在标签中对应的序列号集合
            var confidences = new List<float>();            //可能对象的置信度集合
            var boxes = new List<Rect2d>();                 //可能对象的方框集合

            var w = image.Width;                            //图像的宽高
            var h = image.Height;
            /* 
             YOLO3 COCO 模型输出output的格式：矩阵形式
             其中行代表检测出可能的对象
             其中列为每一个对象的信息，共一个85维[x1,x2,x3,....x85]形式
             列的各个维的信息为：
             0 1 : center 对象框的中心点                   2 3 : w/h  对象框的宽与高
             4 : confidence  对象框的置信度                5 ~ 84 : class probability  每一类的置信度
            */
            const int prefix = 5;                           //分类概率 

            foreach (var item in results)
            {
                for (var i = 0; i < item.Rows; i++)         //取出每一个可能的对象
                {

                    var confidence = item.At<float>(i, 4);                       //第四维：置信度
                    if (confidence > _config.Threshold)
                    {
                        double maxVal, minVal;
                        Point min, max;
                        Cv2.MinMaxLoc(item.Row(i).ColRange(prefix, item.Cols),    //取出5 ~ 84维
                            out minVal, out maxVal, out min, out max);            //求5 ~ 84维中最大的置信度的那个 返回max.X即为其代表的标签序列号

                        var classes = max.X;
                        var probability = item.At<float>(i, classes + prefix);    //取出max.X对应的置信度

                        if (probability > _config.Threshold)                      //more accuracy, you can cancel it
                        {
                            //x,y,width,height 都是相对于输入图片的比例，所以需要乘以相应的宽高进行复原
                            var centerX     = item.At<float>(i, 0) * w;
                            var centerY     = item.At<float>(i, 1) * h;
                            var width       = item.At<float>(i, 2) * w;
                            var height      = item.At<float>(i, 3) * h;
                            var left        = centerX - width / 2;
                            var top         = centerY - height / 2;

                            //准备nms(非极大值抑制)数据
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            boxes.Add(new Rect2d(left, top, width, height));
                        }
                    }
                }
            }

            //nms(非极大值抑制)提取分数最高的
            //去除重叠和低置信度的目标框
            CvDnn.NMSBoxes(boxes, confidences, _config.Threshold, _config.NmsThreshold, out int[] indices);

            foreach (var i in indices)
            {
                //画出目标方框并标注置信度和分类标签
                var box = boxes[i];

                //Build NetResult
                string label        = _config.Labels[classIds[i]];
                double probability  = (double)confidences[i];
                int left            = (int)box.X;
                int top             = (int)box.Y;
                int width           = (int)box.Width;
                int height          = (int)box.Height;
                netResults.Add(NetResult.Add(left, top, width, height, label, probability));

                if (true == _config.IsDraw)
                {
                    Draw(ref image, classIds[i], confidences[i], box.X, box.Y, box.Width, box.Height);
                }

            }

            return netResults.ToArray();        // 返回检测结果
        }

        /// <summary>
        /// 将结果在图像上画出
        /// </summary>
        /// <param name="image"></param>
        /// <param name="classes"> 在标签中的序号</param>
        /// <param name="confidence">置信度</param>
        /// <param name="left">对象框左边距离</param>
        /// <param name="top">对象框顶边距离</param>
        /// <param name="width">对象框宽度</param>
        /// <param name="height">对象框高度</param>
        private void Draw(ref Mat image, int classes, float confidence, double left, double top, double width, double height)
        {
            //标签字符串
            var label = string.Format("{0} {1:0.0}%", _config.Labels[classes], confidence * 100);
            //画方框
            Cv2.Rectangle(image, new Point(left, top), new Point(left + width, top + height), _config.Colors[classes], 1);

            //标签字符大小
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);

            //画标签背景框
            var x1 = left < 0 ? 0 : left;
            Cv2.Rectangle(image, new Rect(new Point(x1, top - textSize.Height - baseline),
                new Size(textSize.Width, textSize.Height + baseline)), _config.Colors[classes], Cv2.FILLED);
            Cv2.PutText(image, label, new Point(x1, top - baseline), HersheyFonts.HersheyTriplex, 0.5, Scalar.White);

            Cv2.ImShow("图片展示：", image);
        }

        /// <summary>
        /// yolo整个处理过程
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        private NetResult[] Process(Mat img)
        {
            Mat blob = ImagePretreatment(img);

            Net net = InitializeModel(_config.ModelWeights, _config.ModelConfiguaration);
            _config.Labels = ReadLabels(_config.LabelsFile);


            net.SetInput(blob);                                    // 输入数据

            var outNames = net.GetUnconnectedOutLayersNames();    //获得输出层名

            var outs = outNames.Select(_ => new Mat()).ToArray(); //转换成 Mat[]或者List<Mat> outputs = new List<Mat>();

            net.Forward(outs, outNames);                          //Execute all out layers

            NetResult[] netResults = Postprocess(ref img, outs);


            return netResults;
        }
        #endregion

        #region demo
        private void Demo()
        {
            string dir                 = System.IO.Directory.GetCurrentDirectory();
            string modelWeights        = System.IO.Path.Combine(dir, "yolov3.weights");
            string modelConfiguaration = System.IO.Path.Combine(dir, "yolov3.cfg");
            string labelFile           = System.IO.Path.Combine(dir, "coconames.txt");
            string imgpath             = System.IO.Path.Combine(dir, "zidane.jpg");
            YoloV3 yoloV3 = new YoloV3(modelWeights, modelConfiguaration, labelFile, 320, 320, 0.5f);
            NetResult[] netResults  = yoloV3.Detect(imgpath);
            foreach (var result in netResults)
            {
                Console.WriteLine("类别，置信度，方框坐标[左边、顶点、长、宽]");
                Console.WriteLine(result.Label + "  " + result.Probability.ToString() + " [" +
                    result.Rectangle.Left.ToString() + " " + result.Rectangle.Top.ToString() + " " + 
                    result.Rectangle.Width.ToString() + " " + result.Rectangle.Height.ToString() + "]");
            }
        }
        #endregion
    }
}
