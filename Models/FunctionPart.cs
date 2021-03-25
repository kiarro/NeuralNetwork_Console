using System;
using System.Collections.Generic;

namespace MatrixNeuralNetwok.Function {
    public static class FunctionPart {
        private static Random rand = new Random(1);

        private static double realMinValue = -1;
        private static double realMaxValue = 1;
        private static double netMinValue = 0;
        // private static double netMinValue = -1;
        private static double netMaxValue = 1;

        /// <summary>
        /// returns value of function
        /// </summary>
        /// <param name="x">x parameter</param>
        /// <param name="y">y parameter</param>
        /// <returns>function value at (x,y) point</returns>
        public static double MyFunc(double x, double y) {
            return Math.Sin(x + y) * Math.Cos(3 * x);
            // return Math.Sin(x)+Math.Sin(y);
        }

        public static CasesSet PrepareCasesRandom(int num, double minX, double maxX, double minY, double maxY) {
            Dictionary<double[], double[]> res = new Dictionary<double[], double[]>();

            for (int i = 0; i < num; i++) {
                double x = rand.NextDouble(minX, maxX);
                double y = rand.NextDouble(minY, maxY);
                res.Add(new double[] { x, y }, new double[] { RealToNet(MyFunc(x, y)) });
            }
            return new CasesSet(res);
        }

        public static CasesSet PrepareCasesUniform(int numX, double minX, double maxX, int numY, double minY, double maxY) {
            Dictionary<double[], double[]> res = new Dictionary<double[], double[]>();
            double dx = (maxX - minX) / (numX - 1);
            double dy = (maxY - minY) / (numY - 1);
            for (double x = minX; x < maxX + dx/10; x+=dx) {
                for (double y = minY; y < maxY + dy/10; y+=dy){
                    res.Add(new double[] { x, y }, new double[] { RealToNet(MyFunc(x, y)) });
                }
            }
            return new CasesSet(res);
        }
        public static double RealToNet(double value) {
            double teta = (value - realMinValue) / (realMaxValue - realMinValue);
            return netMinValue + (netMaxValue - netMinValue) * teta;
        }
        public static double NetToReal(double value) {
            double teta = (value - netMinValue) / (netMaxValue - netMinValue);
            return realMinValue + (realMaxValue - realMinValue) * teta;
        }
        public static void ResetRandom(int seed) {
            rand = new Random(seed);
        }
    }

    static class Ext {
        public static double NextDouble(this Random r, double min, double max) {
            return min + (max - min) * r.NextDouble();
        }
    }
}
