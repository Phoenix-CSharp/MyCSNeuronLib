
namespace Activation.Funcs
{
    public static class Functions
    {
        /// <summary>
        /// Identity activation function (Id)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>x</returns>
        public static double Id(double x) => x;
        /// <summary>
        /// Step activation function (Step)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>1.0 if x >= 0 else 0.0</returns>
        public static double Step(double x) => x >= 0 ? 1.0 : 0.0;
        /// <summary>
        /// Sigmoid activation function (Sigmoid)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>1.0 / (1.0 + e^(-x))</returns>
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        /// <summary>
        /// Sigmoid activation function shorthand (S)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>1.0 / (1.0 + e^(-x))</returns>
        public static double S(double x) => 1.0 / (1.0 + Math.Exp(-x));
        /// <summary>
        /// Hyperbolic Tangent activation function (Tanh)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>th(x)</returns>
        public static double Tanh1(double x) => Math.Tanh(x);
        /// <summary>
        /// Hyperbolic Tangent activation function alternative form (Tanh2)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>(e^x - e^(-x))/(e^x + e^(-x))</returns>
        public static double Tanh2(double x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        /// <summary>
        /// Hyperbolic Tangent activation function shorthand (Th1)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>th(x)</returns>
        public static double Th1(double x) => Math.Tanh(x);
        /// <summary>
        /// Hyperbolic Tangent activation function alternative form shorthand (Th2)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>(e^x - e^(-x))/(e^x + e^(-x))</returns>
        public static double Th2(double x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        /// <summary>
        /// ArcTan activation function (Atan)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>atan(x)</returns>
        public static double Atan(double x) => Math.Atan(x);
        /// <summary>
        /// SoftSign activation function (SoftSing)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>x / (1 + |x|)</returns>
        public static double SoftSing(double x) => x / (1.0 + Math.Abs(x));
        /// <summary>
        /// Inverse Square Root Unit activation function (ISRU)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <param name="A">Hyperparameter responsible for the boundaries of the range of values of the function | E(f(x)) <= |1/ √ A|</param>
        /// <returns>x / √(1 + A * x^2)</returns>
        public static double ISRU(double x, double A) => x / Math.Sqrt(1.0 + A * Math.Pow(x, 2));
        /// <summary>
        /// Rectified Linear Unit activation function (ReLU)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>x if x >= 0 else 0</returns>
        public static double ReLU(double x) => Math.Max(0.0, x);
        /// <summary>
        /// Leaky Rectified Linear Unit activation function (LReLU)
        /// </summary>
        /// <param name="x">Weighted neuron sum</param>
        /// <returns>x if x >= 0 else 0.01*x</returns>
        public static double LReLU(double x) => x >= 0.0 ? x : 0.01 * x;
        public static double 
    }
}