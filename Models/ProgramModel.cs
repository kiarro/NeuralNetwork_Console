using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixNeuralNetwork;
using MatrixNeuralNetwork.FileStorage;

namespace NeuralNetwork_Console.Models
{
    public class Model
    {
        private static readonly NLog.Logger Logger = NLog.LogManager.GetCurrentClassLogger();

        public readonly Dictionary<string, MatrixNN> Networks = new Dictionary<string, MatrixNN>();

        public readonly Dictionary<string, CasesSet> CasesSets = new Dictionary<string, CasesSet>();

        public void CreateNewNet(string name, int[] str)
        {
            if (Networks.ContainsKey(name))
            {
                Console.WriteLine("Network with that name already exists.");
                Logger.Warn("Network with that name already exists.");
            }
            else
            {
                Networks.Add(name, new MatrixNN(str));
            }
        }
        public void ImportNet(string name, string path)
        {
            if (Networks.ContainsKey(name))
            {
                Console.WriteLine("Network with that name already exists.");
                Logger.Warn("Network with that name already exists.");
            }
            else
            {
                Networks.Add(name, ImportExport.ImportNN(path));
            }
        }
        public void ExportNet(string name, string path)
        {
            MatrixNN net;
            bool ex = Networks.TryGetValue(name, out net);
            if (ex)
            {
                ImportExport.ExportNN(net, path);
            }
            else
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
            }
        }
        public void RemoveNet(string name)
        {
            MatrixNN net;
            bool ex = Networks.TryGetValue(name, out net);
            if (ex)
            {
                Networks.Remove(name);
            }
            else
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
            }
        }
        public TrainTask TrainNet(string netN, string casesN, int era = 100, int batchSize = 100, double eduSpeed = 0.3, string resNetName = null)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            CasesSet cases;
            bool exC = CasesSets.TryGetValue(casesN, out cases);
            if (!exN)
            {
                return new TrainTask(Task.Run(() =>
                {
                    Console.WriteLine("Network with that name does not exists.");
                    Logger.Warn("Network with that name does not exists.");
                }), net);
            }
            if (!exC)
            {
                return new TrainTask(Task.Run(() =>
                {
                    Console.WriteLine("Cases set with that name does not exists.");
                    Logger.Warn("Cases set with that name does not exists.");
                }), net);
            }
            if (resNetName is not null)
            {
                if (Networks.ContainsKey(resNetName)) return new TrainTask(Task.Run(() =>
                {
                    Console.WriteLine("Network with new name already exists.");
                    Logger.Warn("Network with new name already exists.");
                }), net);
                net = net.Copy();
                Networks.Add(resNetName, net);
                netN = resNetName;
            }
            return new TrainTask(Task.Run(() =>
            {
                double[] err = net.TrainNet(cases, era, batchSize, eduSpeed);
                // ImportExport.PrintEpochsError(err, String.Format("{0}.EpochStats", netN));
            }), net);
        }
        public double[] EvalValue(string netN, double[] input)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            if (!exN)
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
                return new double[0] { };
            }
            if (net.NetStr[0] != input.Length)
            {
                Console.WriteLine("Wrong count of inputs");
                Logger.Warn("Wrong count of inputs");
                return new double[0] { };
            }
            double[] output = net.EvalValue(input);
            return output;
        }
        public double TestNet(string netN, double[] input, double[] output)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            if (!exN)
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
                return 1000000;
            }
            if (net.NetStr[0] != input.Length)
            {
                Console.WriteLine("Wrong count of inputs");
                Logger.Warn("Wrong count of inputs");
                return 1000000;
            }
            double err = net.TestNet(input, output);
            return err;
        }
        public double TestNet(string netN, string casesN)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            CasesSet cases;
            bool exC = CasesSets.TryGetValue(casesN, out cases);
            if (!exN)
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
                return -1;
            }
            if (!exC)
            {
                Console.WriteLine("Cases set with that name does not exists.");
                Logger.Warn("Cases set with that name does not exists.");
                return -1;
            }
            return net.TestNet(cases);
        }
        public void EvalToFileNet(string netN, string casesN, string path)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            CasesSet cases;
            bool exC = CasesSets.TryGetValue(casesN, out cases);
            if (!exN)
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
                return;
            }
            if (!exC)
            {
                Console.WriteLine("Cases set with that name does not exists.");
                Logger.Warn("Cases set with that name does not exists.");
                return;
            }
            CasesSet res = net.EvalSetNet(cases);
            ImportExport.ExportCasesSet(res, path);
        }

        public void CreateTestCases(string name)
        {
            if (CasesSets.ContainsKey(name))
            {
                Console.WriteLine("Cases set with that name already exists.");
                Logger.Warn("Cases set with that name already exists.");
            }
            else
            {
                Dictionary<double[], double[]> res = new Dictionary<double[], double[]>();

                res.Add(new double[] {0, 0}, new double[] {0, 0, 0});
                res.Add(new double[] {0, 1}, new double[] {0, 1, 1});
                res.Add(new double[] {1, 0}, new double[] {0, 1, 1});
                res.Add(new double[] {1, 1}, new double[] {1, 1, 0});

                CasesSets.Add(name, new CasesSet(res));
            }
        }
        public void ImportCasesSet(string name, string path)
        {
            if (CasesSets.ContainsKey(name))
            {
                Console.WriteLine("Cases set with that name already exists.");
                Logger.Warn("Cases set with that name already exists.");
            }
            else
            {
                CasesSets.Add(name, ImportExport.ImportCasesSet(path));
            }
        }
        public void ExportCasesSet(string name, string path)
        {
            CasesSet cs;
            bool ex = CasesSets.TryGetValue(name, out cs);
            if (CasesSets.ContainsKey(name))
            {
                ImportExport.ExportCasesSet(cs, path);
            }
            else
            {
                Console.WriteLine("Cases set with that name does not exists.");
                Logger.Warn("Cases set with that name does not exists.");
            }
        }
        public void RemoveCasesSet(string name)
        {
            if (CasesSets.ContainsKey(name))
            {
                CasesSets.Remove(name);
            }
            else
            {
                Console.WriteLine("Cases set with that name does not exists.");
                Logger.Warn("Cases set with that name does not exists.");
            }
        }
        public void CopyNet(string name1, string name2)
        {
            MatrixNN net;
            bool exN = Networks.TryGetValue(name1, out net);
            if (Networks.ContainsKey(name2))
            {
                Console.WriteLine("Network with that name already exists.");
                Logger.Warn("Network with that name already exists.");
            }
            else if (!exN)
            {
                Console.WriteLine("Network with that name does not exists.");
                Logger.Warn("Network with that name does not exists.");
            }
            else
            {
                net = net.Copy();
                Networks.Add(name2, net);
            }
        }
    }

    public class TrainTask
    {
        public Task CurrentTask { get; }
        MatrixNN Network { get; }
        public int Era { get => Network.CurrentEra; }
        public int Element { get => Network.CurrentElement; }
        public int EraCount { get => Network.CountEra; }
        public int ElementCount { get => Network.CountElement; }
        public TrainTask(Task task, MatrixNN net)
        {
            CurrentTask = task;
            Network = net;
        }
    }
}
