using System;

namespace Activation.Funcs
{
    public static class Func
    {
        private static Random? _rnd;
        private static double? _max_base;

        public static double Id(double x) => x;
        public static double Step(double x) => x >= 0 ? 1.0 : 0.0;
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double S(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double Tanh1(double x) => Math.Tanh(x);
        public static double Tanh2(double x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        public static double Th1(double x) => Math.Tanh(x);
        public static double Th2(double x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        public static double Atan(double x) => Math.Atan(x);
        public static double SoftSing(double x) => x / (1.0 + Math.Abs(x));
        public static double ISRU(double x, double A) => x / Math.Sqrt(1.0 + A * Math.Pow(x, 2));
        public static double ReLU(double x) => Math.Max(0.0, x);
        public static double LReLU(double x) => x >= 0 ? x : 0.01 * x;
        public static double PReLU(double x, double A) => x >= 0 ? x : A * x;
        public static double RReLU(double x, double highBound) {
            _rnd ??= new Random();
            _max_base ??= Math.Max(0.0, highBound);
            var A = _rnd.NextDouble() * (_max_base - 0.001) + 0.001; // [0.001..highBound]
            return x >= 0 ? x : (double)A* x;
        }
        public static double ELU(double x, double A) => x >= 0 ? x : A * (Math.Exp(x) - 1);
        public static double SELU(double x, double A = 1.67326) => 1.0507 * ELU(x, A);
        public static double SReLU(double x, double[] param)
        {
            if (param.Length != 4) throw new ArgumentException("This array is not a suitable shape, suitable: [t_l, a_l, t_r, a_r]");
            if (x <= param[0]) return param[0] + param[1] * (x - param[0]);
            else if (param[0]< x && x < param[2]) return x;
            else return param[2] + param[3] * (x - param[2]);
        }
        public static double ISRLU(double x, double A) => x >= 0.0 ? x : ISRU(x, A);
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
        public static double SoftPlus(double x) => Math.Log(1.0 + Math.Exp(x));
        public static double BId(double x) => (Math.Sqrt(Math.Pow(x,2) + 1) - 1) / 2 + x;
        public static double SiLU(double x)
        {
            Func<double, double> S = (x) => 1.0 / (1.0 + Math.Exp(-x)); 
            return x * S(x);
        }
        public static double SoftExponential(double x, double A)
        {
            if (A < 0) return -Math.Log(1 - A * (x + A)) / A;
            else if (A == 0) return x;
            else return (Math.Exp(A * x) - 1) / A + A;
        }
        public static double Sinusoid(double x) => Math.Sin(x);
        public static double Sinc(double x) => x == 0 ? 1.0 : Math.Sin(x) / x;
        public static double Gaussin(double x) => Math.Exp(-Math.Pow(x, 2));

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
        public static double Maxout(double[] x)
        {
            var max = double.NegativeInfinity;
            for (int i = 0; i < x.Length; i++)
            {
                max = Math.Max(max, x[i]);
            }
            return max;
        }
   }
}