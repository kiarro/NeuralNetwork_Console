using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace MatrixNeuralNetwok {
    public class MatrixNN {
        public string Structure {
            get {
                StringBuilder str = new StringBuilder("{");
                foreach (Vector<double> layer in Neurons) {
                    str.AppendFormat("{0},", layer.Count);
                }
                str.Remove(str.Length - 1, 1);
                str.Append("}\n");
                return str.ToString();
            }
        }
        public int LayerAmount { get; private set; }
        public Vector<double>[] Neurons { get; }
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
            LayerAmount = NNstructure.Length;
            Neurons = new Vector<double>[LayerAmount];
            dNeurons = new Vector<double>[LayerAmount];
            Shift = new Vector<double>[LayerAmount];
            dShift = new Vector<double>[LayerAmount];
            Weights = new Matrix<double>[LayerAmount - 1];
            dWeights = new Matrix<double>[LayerAmount - 1];
            for (int i = 0; i < LayerAmount - 1; i++) {
                Neurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dNeurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                Weights[i] = Matrix<double>.Build.Random(NNstructure[i + 1], NNstructure[i]);
                // Weights[i] = Matrix<double>.Build.Dense(NNstructure[i+1], NNstructure[i], 0.1);
                dWeights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], 0);
                Shift[i] = Vector<double>.Build.Random(NNstructure[i + 1], NNstructure[i]);
                dShift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], 0);
            }
            Neurons[LayerAmount - 1] = Vector<double>.Build.Dense(NNstructure[LayerAmount - 1], 0);
        }

        public MatrixNN(int[] NNstructure, double? value) {
            LayerAmount = NNstructure.Length;
            Neurons = new Vector<double>[LayerAmount];
            dNeurons = new Vector<double>[LayerAmount];
            Shift = new Vector<double>[LayerAmount];
            dShift = new Vector<double>[LayerAmount];
            Weights = new Matrix<double>[LayerAmount - 1];
            dWeights = new Matrix<double>[LayerAmount - 1];

            for (int i = 0; i < LayerAmount - 1; i++) {
                Neurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dNeurons[i] = Vector<double>.Build.Dense(NNstructure[i], 0);
                dWeights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], 0);
                dShift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], 0);
            }
            Neurons[LayerAmount - 1] = Vector<double>.Build.Dense(NNstructure[LayerAmount - 1], 0);

            if (value is not null) {
                for (int i = 0; i < LayerAmount - 1; i++) {
                    Weights[i] = Matrix<double>.Build.Dense(NNstructure[i + 1], NNstructure[i], (double)value);
                    Shift[i] = Vector<double>.Build.Dense(NNstructure[i + 1], (double)value);
                }
                Neurons[LayerAmount - 1] = Vector<double>.Build.Dense(NNstructure[LayerAmount - 1], 0);
            }
        }

        private double ActivationFunction(double value) {
            return 1 / (1 + Math.Exp(-value));
            // return value;
        }
        private double ActFuncDiff(double power) {
            return power * (1 - power);
            // return 1;
        }

        private double[] ForwardPass(double[] inputs) {
            Neurons[0] = Vector<double>.Build.DenseOfArray(inputs);
            for (int i = 1; i < LayerAmount; i++) {
                Neurons[i] = (Weights[i - 1] * Neurons[i - 1] + Shift[i - 1]).Map(ActivationFunction);
            }
            return Neurons[LayerAmount - 1].AsArray();
        }

        private double TrainCaseBackpropagation(double[] input, double[] idealOutput, double eduSpeed) {
            double[] output = ForwardPass(input);
            // find error
            double error = 0;
            for (int i = 0; i < output.Length; i++) {
                error += (output[i] - idealOutput[i]) * (output[i] - idealOutput[i]);
            }
            // go backward
            // count deltas
            dNeurons[LayerAmount - 1] = (Neurons[LayerAmount - 1] - Vector<double>.Build.DenseOfArray(idealOutput)).PointwiseMultiply(Neurons[LayerAmount - 1]).PointwiseMultiply(1 - Neurons[LayerAmount - 1]);
            for (int i = LayerAmount - 2; i > 0; i--) {
                dNeurons[i] = Neurons[i].PointwiseMultiply(1 - Neurons[i]).PointwiseMultiply(Weights[i].Transpose() * dNeurons[i + 1]);
            }
            // count weight changes
            for (int i = 0; i < LayerAmount - 1; i++) {
                dWeights[i] += -eduSpeed * dNeurons[i + 1].OuterProduct(Neurons[i]);
                dShift[i] += -eduSpeed * dNeurons[i + 1] * 1;
                // change weights in outer function for batched training
            }

            return error;
        }

        public void TrainNet(CasesSet trainSet, int eraAmount = 10000, int batchSize = 1, double eduSpeed = 0.3) {
            CountElement = trainSet.Count;
            CountEra = eraAmount;
            int caseInBatch = 0;
            double meanError = 0;
            for (era = 0; era < eraAmount; era++) {
                meanError = 0;
                el = 0;
                // Parallel.ForEach(trainSet, item => {
                //     meanError += TrainCaseBackpropagation(item.Key, item.Value, eduSpeed);
                //     caseInBatch++;
                //     el++;
                //     if (caseInBatch == batchSize) {
                //         caseInBatch = 0;
                //         for (int i = 0; i < LayerAmount - 1; i++) {
                //             Weights[i] = Weights[i] + dWeights[i];
                //             Shift[i] = Shift[i] + dShift[i];
                //             dWeights[i].Clear();
                //             dShift[i].Clear();
                //         }
                //     }
                // });
                foreach (var item in trainSet)
                {
                    meanError += TrainCaseBackpropagation(item.Key, item.Value, eduSpeed);
                    caseInBatch++; el++;
                    if (caseInBatch == batchSize)
                    {
                        caseInBatch = 0;
                        for (int i = 0; i < LayerAmount - 1; i++)
                        {
                            Weights[i] = Weights[i] + dWeights[i];
                            Shift[i] = Shift[i] + dShift[i];
                            dWeights[i].Clear();
                            dShift[i].Clear();
                        }
                    }
                }
                caseInBatch = 0;
                for (int i = 0; i < LayerAmount - 1; i++) {
                    Weights[i] = Weights[i] + dWeights[i];
                    Shift[i] = Shift[i] + dShift[i];
                    dWeights[i].Clear();
                    dShift[i].Clear();
                }
            }
        }

        public double TestNet(CasesSet testSet) {
            double error = 0;
            double errorMean = 0;
            double[] output;
            foreach (var item in testSet) {
                output = ForwardPass(item.Key);
                for (int i = 0; i < output.Length; i++) {
                    error += (output[i] - item.Value[i]) * (output[i] - item.Value[i]);
                }
                errorMean += error;
            }
            errorMean /= testSet.Count;
            return errorMean;
        }

    }
}
