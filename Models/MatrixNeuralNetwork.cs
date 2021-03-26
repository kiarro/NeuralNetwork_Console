using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace MatrixNeuralNetwok {
    public class MatrixNN {
        public string Structure {
            get {
                StringBuilder str = new StringBuilder("{");
                foreach (int layerc in NetStr) {
                    str.AppendFormat("{0},", layerc);
                }
                str.Remove(str.Length - 1, 1);
                str.Append("}\n");
                return str.ToString();
            }
        }
        public int LayerAmount { get; private set; }
        public int[] NetStr { get; private set; }
        // public Vector<double>[] Neurons { get; }
        private Vector<double>[] dNeurons { get; set; }
        public Matrix<double>[] Weights { get; set; }
        private Matrix<double>[] dWeights { get; set; }
        public Vector<double>[] Shift { get; set; }
        private Vector<double>[] dShift { get; set; }

        public int CurrentEra { get => era; }
        private int era;
        public int CurrentElement { get => el; }
        private int el;
        public int CountEra { get; private set; }
        public int CountElement { get; private set; }

        public MatrixNN(int[] NNstructure) {
            NetStr = NNstructure;
            LayerAmount = NNstructure.Length;
            // Neurons = new Vector<double>[LayerAmount];
            dNeurons = new Vector<double>[LayerAmount];
            Shift = new Vector<double>[LayerAmount];
            dShift = new Vector<double>[LayerAmount];
            Weights = new Matrix<double>[LayerAmount - 1];
            dWeights = new Matrix<double>[LayerAmount - 1];
            for (int i = 0; i < LayerAmount - 1; i++) {
                // Neurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dNeurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                Weights[i] = Matrix<double>.Build.Random(NNstructure[i + 1], NNstructure[i]);
                // Weights[i] = Matrix<double>.Build.Dense(NNstructure[i+1], NNstructure[i], 0.1);
                dWeights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], 0);
                Shift[i] = Vector<double>.Build.Random(NNstructure[i + 1], NNstructure[i]);
                dShift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], 0);
            }
            // Neurons[LayerAmount - 1] = Vector<double>.Build.Dense(NNstructure[LayerAmount - 1], 0);
        }

        public MatrixNN(int[] NNstructure, double? value) {
            LayerAmount = NNstructure.Length;
            // Neurons = new Vector<double>[LayerAmount];
            dNeurons = new Vector<double>[LayerAmount];
            Shift = new Vector<double>[LayerAmount];
            dShift = new Vector<double>[LayerAmount];
            Weights = new Matrix<double>[LayerAmount - 1];
            dWeights = new Matrix<double>[LayerAmount - 1];

            for (int i = 0; i < LayerAmount - 1; i++) {
                // Neurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dNeurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dWeights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], 0);
                dShift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], 0);
            }
            // Neurons[LayerAmount - 1] = Vector<double>.Build.Dense(NNstructure[LayerAmount - 1], 0);

            if (value is not null) {
                for (int i = 0; i < LayerAmount - 1; i++) {
                    Weights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], (double)value);
                    Shift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], (double)value);
                }
            }
        }

        private double ActivationFunction(double value) {
            return 1 / (1 + Math.Exp(-value));
            // return Math.Sin(value);
            // return value;
        }
        private double ActFuncDiff(double power) {
            return power * (1 - power);
            // return Math.Sqrt(1-power*power);
            // return 1;
        }

        private Vector<double>[] ForwardPass(double[] inputs) {
            Vector<double>[] Neurons = new Vector<double>[LayerAmount];
            Neurons[0] = Vector<double>.Build.DenseOfArray(inputs);
            for (int i = 1; i < LayerAmount; i++) {
                Neurons[i] = (Weights[i - 1] * Neurons[i - 1] + Shift[i - 1]).Map(ActivationFunction);
            }
            return Neurons;
        }

        public double[] EvalValue(double[] inputs) {
            return ForwardPass(inputs)[LayerAmount - 1].AsArray();
        }

        private ValueTriple<double, Matrix<double>[], Vector<double>[]> TrainCaseBackpropagation(double[] input, double[] idealOutput, double eduSpeed) {
            Vector<double>[] Neurons = ForwardPass(input);
            double[] output = Neurons[LayerAmount - 1].AsArray();
            // find error
            double error = 0;
            for (int i = 0; i < output.Length; i++) {
                error += (output[i] - idealOutput[i]) * (output[i] - idealOutput[i]);
            }
            // go backward
            // count deltas
            Vector<double>[] dN = new Vector<double>[LayerAmount];
            dN[LayerAmount - 1] = (Neurons[LayerAmount - 1] - Vector<double>.Build.DenseOfArray(idealOutput)).PointwiseMultiply(Neurons[LayerAmount - 1]).PointwiseMultiply(1 - Neurons[LayerAmount - 1]);
            for (int i = LayerAmount - 2; i > 0; i--) {
                dN[i] = Neurons[i].PointwiseMultiply(1 - Neurons[i]).PointwiseMultiply(Weights[i].Transpose() * dN[i + 1]);
            }
            // count weight changes
            Matrix<double>[] dW = new Matrix<double>[LayerAmount - 1];
            Vector<double>[] dS = new Vector<double>[LayerAmount - 1];
            for (int i = 0; i < LayerAmount - 1; i++) {
                dW[i] = -eduSpeed * dN[i + 1].OuterProduct(Neurons[i]);
                dS[i] = -eduSpeed * dN[i + 1] * 1;
                // change weights in outer function for batched training
            }

            return new ValueTriple<double, Matrix<double>[], Vector<double>[]>(error, dW, dS);
        }

        public double[] TrainNet(CasesSet trainSet, int eraAmount = 10000, int batchSize = 1, double eduSpeed = 0.3) {
            CountElement = trainSet.Count;
            CountEra = eraAmount;
            double[] meanError = new double[eraAmount];
            int counter = 0;
            var batches = trainSet.GroupBy(x => (int)(counter++ / batchSize));
            for (era = 0; era < eraAmount; era++) {
                meanError[era] = 0;
                el = 0;
                foreach (var batch in batches) {
                    foreach (var item in batch) {
                        ValueTriple<double, Matrix<double>[], Vector<double>[]> v = TrainCaseBackpropagation(item.Key, item.Value, eduSpeed);
                        meanError[era] += v.Value1;
                        for (int i = 0; i < LayerAmount - 1; i++) {
                            dWeights[i] += v.Value2[i];
                            dShift[i] += v.Value3[i];
                        }
                        el++;
                    }
                    for (int i = 0; i < LayerAmount - 1; i++) {
                        Weights[i] = Weights[i] + dWeights[i];
                        Shift[i] = Shift[i] + dShift[i];
                        dWeights[i].Clear();
                        dShift[i].Clear();
                    }
                }
                meanError[era] /= CountElement;
            }
            return meanError;
        }

        public double TestNet(CasesSet testSet) {
            double error = 0;
            double errorMean = 0;
            double[] output;
            foreach (var item in testSet) {
                output = ForwardPass(item.Key)[LayerAmount - 1].ToArray();
                error = 0;
                for (int i = 0; i < output.Length; i++) {
                    error += (output[i] - item.Value[i]) * (output[i] - item.Value[i]);
                }
                errorMean += error;
            }
            errorMean /= testSet.Count;
            return errorMean;
        }

        public double TestNet(double[] input, double[] output) {
            double error = 0;
            double[] out1 = ForwardPass(input)[LayerAmount - 1].AsArray();
            for (int i = 0; i < output.Length; i++) {
                error += (out1[i] - output[i]) * (out1[i] - output[i]);
            }
            return error;
        }

        struct ValueTriple<T, G, H> {
            public T Value1 { get; set; }
            public G Value2 { get; set; }
            public H Value3 { get; set; }
            public ValueTriple(T v1, G v2, H v3) {
                Value1 = v1;
                Value2 = v2;
                Value3 = v3;
            }
        }

        public MatrixNN Copy() {
            MatrixNN res = new MatrixNN(NetStr);
            for (int i = 0; i < LayerAmount - 1; i++) {
                res.Weights[i] = Weights[i];
                res.Shift[i] = Shift[i];
            }
            return res;
        }

    }
}
