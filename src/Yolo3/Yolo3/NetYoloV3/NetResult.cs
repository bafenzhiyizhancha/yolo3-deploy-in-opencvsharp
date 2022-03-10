
using System.Drawing;

namespace NetYoloV3
{

    /// <summary>
    /// 结果存储格式
    /// </summary>
    public class NetResult
    {
        /// <summary>
        /// Bounding Box  方框
        /// </summary>
        public Rectangle Rectangle { get; set; }

        /// <summary>
        /// 置信度
        /// </summary>
        public double Probability { get; set; }

        /// <summary>
        /// 类别
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// 添加
        /// </summary>
        /// <param name="left"> 方框的左边 </param>
        /// <param name="top">方框的顶点</param>
        /// <param name="width">方框的宽度</param>
        /// <param name="height">方框的高度</param>
        /// <param name="label">对应的类别</param>
        /// <param name="probability">置信度</param>
        /// <returns></returns>
        internal static NetResult Add(int left, int top, int width, int height, string label, double probability)
        {
            return new NetResult()
            {
                Label = label,
                Probability = probability,
                Rectangle = new Rectangle(left, top, width, height)
            };
        }
    }
}
