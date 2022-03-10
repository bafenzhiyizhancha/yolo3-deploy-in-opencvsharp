using NetYoloV3;
using System;
using System.Windows;


namespace Yolo3
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Init();
        }

        private void Init()
        {

            string dir              = System.IO.Directory.GetCurrentDirectory();
            string path             = dir + "\\model\\";
            string image            = dir + "\\image\\";
            string modelWeights     = System.IO.Path.Combine(path, "best.weights");
            string modelConfiguaration = System.IO.Path.Combine(path, "yolov3-tiny.cfg");
            string labelFile        = System.IO.Path.Combine(path, "yolo.names");
            string imgpath          = System.IO.Path.Combine(image, "10000011.bmp");


            YoloV3 yoloV3 = new YoloV3(modelWeights, modelConfiguaration, labelFile, 416, 416, 0.2f);
            NetResult[] netResults = yoloV3.Detect(imgpath);

            foreach (var result in netResults)
            {
                Console.WriteLine("类别，置信度，方框坐标[左边、顶点、长、宽]");
                Console.WriteLine(result.Label + "  " + result.Probability.ToString() + " [" +
                    result.Rectangle.Left.ToString() + " " + result.Rectangle.Top.ToString() + " " +
                    result.Rectangle.Width.ToString() + " " + result.Rectangle.Height.ToString() + "]");
            }
        }
    }
}
