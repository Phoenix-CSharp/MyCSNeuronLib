using System;
using System.Diagnostics.Contracts;

namespace Activation.Derivs
{
    public static class Deriv
    {
        private static Random? _rnd;
        private static double? _rnd_base;

        public static double Havyside(double x) => x >= 0 ? 1.0 : 0.0;
        public static double Havyside(double x, double zero_val) => x == 0 ? zero_val : (x >= 0 ? 1.0 : 0.0);
        public static double H(double x) => Havyside(x);
        public static double H(double x, double zero_val) => Havyside(x, zero_val);
        public static double Id(double x) => 1.0;
        public static double Step(double x, double threshold = 0.0) => x == 0 ? 0.0 : threshold;
        public static double Sigmoid(double x)
        {
            Func<double, double> f = z => 1.0 / (1.0 + Math.Exp(-z));
            return f(x) * (1.0 - f(x));
        }
        public static double S(double x)
        {
            Func<double, double> f = z => 1.0 / (1.0 + Math.Exp(-z));
            return f(x) * (1.0 - f(x));
        }
        public static double Tanh(double x)
        {
            Func<double, double> f = z => Math.Tanh(z);
            return 1.0 - Math.Pow(f(x), 2);
        }
        public static double Th(double x) => Tanh(x);
        public static double Atan(double x) => 1.0 / (1.0 + Math.Pow(x, 2));
        public static double SoftSing(double x) => 1.0 / Math.Pow(1.0 + Math.Abs(x), 2);
        public static double ISRU(double x, double A)
        {
            Func<double, double> f = z => 1 / Math.Sqrt(1.0 + A * Math.Pow(z, 2));
            return Math.Pow(f(x), 3);
        }
        public static double ReLU(double x) => x >= 0 ? 1.0 : 0.0;
        public static double LReLU(double x) => x >= 0 ? 1.0 : 0.01;
        public static double PReLU(double x, double A) => x >= 0 ? 1.0 : A;
        public static double RReLU(double x, double highBound)
        {
            _rnd ??= new Random();
            _rnd_base = highBound;
            var A = _rnd.NextDouble() * (_rnd_base - 0.001) + 0.001; // A is [0.001, highBound]
            return x >= 0 ? 1.0 : (double)A;
        }
        
    }
}