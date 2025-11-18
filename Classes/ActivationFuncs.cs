using System;
using System.Linq;

namespace Activation.Funcs
{
    /// <summary>
    /// A collection of common activation functions used in neural networks.
    /// </summary>
    public static class Func
    {
        /// <summary>
        /// Random number generator for RReLU function.
        /// </summary>
        private static Random? _rnd;
        /// <summary>
        /// Maximum base value for RReLU function.
        /// </summary>
        private static double? _max_base;
        /// <summary>
        /// Identity activation function (Id).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>x</returns>
        public static double Id(double x) => x;
        /// <summary>
        /// Step activation function (Step).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>1.0 if x &gt;= 0 else 0.0</returns>
        public static double Step(double x) => x >= 0 ? 1.0 : 0.0;
        /// <summary>
        /// Sigmoid activation function (Sigmoid).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>1 / (1 + e^(-x))</returns>
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        /// <summary>
        /// Sigmoid activation function shortword (S).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>1 / (1 + e^(-x))</returns>
        public static double S(double x) => 1.0 / (1.0 + Math.Exp(-x));
        /// <summary>
        /// Hyperbolic tangent activation function (Tanh1).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>Tanh(x)</returns>
        public static double Tanh1(double x) => Math.Tanh(x);
        /// <summary>
        /// Hyperbolic tangent activation function alternative form (Tanh2).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>(e^x - e^(-x)) / (e^x + e^(-x))</returns>
        public static double Tanh2(double x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        /// <summary>
        /// Hyperbolic tangent activation function shortword (Th1).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>Tanh(x)</returns>
        public static double Th1(double x) => Tanh1(x);
        /// <summary>
        /// Hyperbolic tangent activation function shortword alternative form (Th2).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>(e^x - e^(-x)) / (e^x + e^(-x))</returns>
        public static double Th2(double x) => Tanh2(x);
        /// <summary>
        /// Arc tangent activation function (Atan).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>Atan(x)</returns>
        public static double Atan(double x) => Math.Atan(x);
        /// <summary>
        /// SoftSign activation function (SoftSing).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>x / (1 + |x|)</returns>
        public static double SoftSing(double x) => x / (1.0 + Math.Abs(x));
        /// <summary>
        /// Inverse Square Root Unit activation function (ISRU).
        /// </summary>
        /// <param name="x">Weighted sum </param>
        /// <param name="A">Hyperparameter responsible for the boundaries of the range of values of the function E(f(x)) &lt;= |1 / &#8730;A|</param>
        /// <returns> x / &#8730;(1 + A * x^2) </returns>
        public static double ISRU(double x, double A) => x / Math.Sqrt(1.0 + A * Math.Pow(x, 2));
        /// <summary>
        /// Rectified Linear Unit activation function (ReLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns> Max(0, x) </returns>
        public static double ReLU(double x) => Math.Max(0.0, x);
        /// <summary>
        /// Leaky Rectified Linear Unit activation function (LReLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>x if x &gt;= 0 else 0.01 * x</returns>
        public static double LReLU(double x) => x >= 0 ? x : 0.01 * x;
        /// <summary>
        /// Parametric Rectified Linear Unit activation function (PReLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="A">Hyperparameter responsible for the slope velocity to the left of zero</param>
        /// <returns>x if x &gt;= 0 else A * x</returns>
        public static double PReLU(double x, double A) => x >= 0 ? x : A * x;
        /// <summary>
        /// Randomized Rectified Linear Unit activation function (RReLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="highBound">The value of the upper bound for setting the value of parameter A, where A is at [0.001, highBound]</param>
        /// <returns>x if x &gt;= 0 else A * x</returns>
        public static double RReLU(double x, double highBound) {
            _rnd ??= new Random();
            _max_base ??= Math.Max(0.0, highBound);
            var A = _rnd.NextDouble() * (_max_base - 0.001) + 0.001; // [0.001..highBound]
            return x >= 0 ? x : (double)A* x;
        }
        /// <summary>
        /// Exponential Linear Unit activation function (ELU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="A">Hyperparameter responsible for the slope velocity to the left of zero</param>
        /// <returns>x if x &gt;= 0 else A * (e^x - 1)</returns>
        public static double ELU(double x, double A) => x >= 0 ? x : A * (Math.Exp(x) - 1);
        /// <summary>
        /// Scaled Exponential Linear Unit activation function (SELU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="A">Hyperparameter responsible for the slope velocity to the left of zero</param>
        /// <returns>L * x if x &gt;= 0 else L * A * (e^x - 1) | L = 1.0507</returns>
        public static double SELU(double x, double A = 1.67326) => 1.0507 * ELU(x, A);
        /// <summary>
        /// Soft Rectified Linear Unit activation function (SReLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="param">An array of parameters consisting of 4 parameters of the form [t_l, a_l, t_r, a_r]</param>
        /// <returns>t_l + a_l * (x - t_l) if x &lt;= t_l else x if t_l &lt; x &lt; t_r else t_r + a_r * (x - t_r)</returns>
        /// <exception cref="ArgumentException">This exception occurs if the array of parameters does not have the proper form, namely [t_l, a_l, t_r, a_r]</exception>
        public static double SReLU(double x, double[] param)
        {
            if (param.Length != 4) throw new ArgumentException("This array is not a suitable shape, suitable: [t_l, a_l, t_r, a_r]");

            if (x <= param[0]) return param[0] + param[1] * (x - param[0]);
            else if (param[0]< x && x < param[2]) return x;
            else return param[2] + param[3] * (x - param[2]);
        }
        /// <summary>
        /// Inverse Square Root Linear Unit activation function (ISRLU).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="A">Hyperparameter responsible for the boundaries of the range of values of the function E(f(x)) &lt;= |1 / &#8730;A|</param>
        /// <returns>x if x &gt;= 0 else ISRU(x, A) | <see cref="ISRU">see ISRU</see></returns>
        public static double ISRLU(double x, double A) => x >= 0.0 ? x : ISRU(x, A);
        /// <summary>
        /// Adaptive Piecewise Linear Unit activation function (APL).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="a">array of angular coefficients of the slope of a straight line on a given section</param>
        /// <param name="b">An array of function values for a given section</param>
        /// <returns>ReLU(x) + Sum(a[i]*Max(0, -x + b[i]))</returns>
        /// <exception cref="ArgumentException">This exception occurs if arrays a and b do not have equal lengths.</exception>
        public static double APL(double x, double[] a, double[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException("Arrays a and b must have equal lengths.");
            var res = ReLU(x);
            for (int i = 0; i < a.Length; i++)
            {
                res += a[i] * Math.Max(0.0, -x + b[i]);
            }
            return res;
        }
        /// <summary>
        /// SoftPlus activation function (SoftPlus).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>ln(1 + e^x)</returns>
        public static double SoftPlus(double x) => Math.Log(1.0 + Math.Exp(x));
        /// <summary>
        /// Bent Identity activation function (BId).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>(&#8730;(x^2 + 1) - 1) / 2 + x</returns>
        public static double BId(double x) => (Math.Sqrt(Math.Pow(x,2) + 1) - 1) / 2 + x;
        /// <summary>
        /// Sigmoid Linear Unit activation function (SiLU).
        /// <code>
        ///     S(X) = Sigmoid(X)
        /// </code>
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>x * S(x)</returns>
        public static double SiLU(double x) => x * S(x);
        /// <summary>
        /// Soft Exponential activation function (SoftExponential).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <param name="A">The hyperparameter responsible for the shape of the curve</param>
        /// <returns>-ln(1 - A * (x + A)) / A if A &lt; 0 else x if A == 0 else (e^(A * x) - 1) / A + A</returns>
        public static double SoftExponential(double x, double A)
        {
            if (A < 0) return -Math.Log(1 - A * (x + A)) / A;
            else if (A == 0) return x;
            else return (Math.Exp(A * x) - 1) / A + A;
        }
        /// <summary>
        /// Sinusoid activation function (Sinusoid).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>Sin(x)</returns>
        public static double Sinusoid(double x) => Math.Sin(x);
        /// <summary>
        /// Sinc activation function (Sinc).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>1 if x == 0 else Sin(x) / x</returns>
        public static double Sinc(double x) => x == 0 ? 1.0 : Math.Sin(x) / x;
        /// <summary>
        /// Gaussin activation function (Gaussin).
        /// </summary>
        /// <param name="x">Weighted sum</param>
        /// <returns>e^(-(x^2))</returns>
        public static double Gaussin(double x) => Math.Exp(-Math.Pow(x, 2));
        /// <summary>
        /// Softmax activation function (Softmax).
        /// </summary>
        /// <param name="x">vector of raw real estimates (logits)</param>
        /// <returns>probability vector</returns>
        public static double[] Softmax(double[] x)
        {
            double max = x.Max();

            double[] exp = new double[x.Length];
            double sum = 0;

            for (int i = 0; i < x.Length; i++)
            {
                exp[i] = Math.Exp(x[i] - max);
                sum += exp[i];
            }

            for (int i = 0; i < x.Length; i++)
                exp[i] /= sum;

            return exp;
        }
        /// <summary>
        /// Maxout activation function (Maxout).
        /// </summary>
        /// <param name="x">Neuron uotput values</param>
        /// <returns>Max(x)</returns>
        public static double Maxout(double[] x) => x.Max();
   }
}