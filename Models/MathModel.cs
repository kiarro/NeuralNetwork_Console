using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixNeuralNetwok;
using MatrixNeuralNetwok.FileStorage;
using MatrixNeuralNetwok.Function;

namespace NeuralNetwork_Console.Models {
    public class MathModel {
        // public class NetworkWrap {
        //     internal MatrixNN Network { get; }
        //     public string Name { get; private set; }
        //     public NetworkWrap(MatrixNN network, string name) {
        //         this.Network = network;
        //         Name = name;
        //     }
        //     public string Structure { get => Network.Structure; }
        //     public void TrainNet(CasesWrap cases) {
        //         Network.TrainNet(cases.Cases);
        //     }
        //     public double TestNet(CasesWrap cases) {
        //         return Network.TestNet(cases.Cases);
        //     }
        // }
        // public class CasesWrap {
        //     internal CasesSet Cases { get; }
        //     public string Name { get; }
        //     public CasesWrap(CasesSet cases, string name) {
        //         Cases = cases;
        //         Name = name;
        //     }
        // }

        public readonly Dictionary<string, MatrixNN> Networks = new Dictionary<string, MatrixNN>();

        public readonly Dictionary<string, CasesSet> CasesSets = new Dictionary<string, CasesSet>();

        public void SetRandomSeed(int seed) {
            FunctionPart.ResetRandom(seed);
        }
        public void CreateNewNet(string name, int[] str) {
            if (Networks.ContainsKey(name))
                Console.WriteLine("Network with that name already exists.");
            else
                Networks.Add(name, new MatrixNN(str));
        }
        public void ImportNet(string name, string path) {
            if (Networks.ContainsKey(name))
                Console.WriteLine("Network with that name already exists.");
            else
                Networks.Add(name, ImportExport.ImportNN(path));
        }
        public void ExportNet(string name, string path) {
            MatrixNN net;
            bool ex = Networks.TryGetValue(name, out net);
            if (ex)
                ImportExport.ExportNN(net, path);
            else
                Console.WriteLine("Network with that name does not exists.");
        }
        public void RemoveNet(string name) {
            MatrixNN net;
            bool ex = Networks.TryGetValue(name, out net);
            if (ex)
                Networks.Remove(name);
            else
                Console.WriteLine("Network with that name does not exists.");
        }
        public TrainTask TrainNet(string netN, string casesN, int era = 10000, int batchSize = 100, double eduSpeed = 0.3) {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            CasesSet cases;
            bool exC = CasesSets.TryGetValue(casesN, out cases);
            if (!exN) {
                return new TrainTask(Task.Run(()=>Console.WriteLine("Network with that name does not exists.")), net);
            }
            if (!exC) {
                return new TrainTask(Task.Run(()=>Console.WriteLine("Cases set with that name does not exists.")), net);
            }
            return new TrainTask(Task.Run(()=>net.TrainNet(cases, era, batchSize, eduSpeed)), net);
        }
        public double TestNet(string netN, string casesN) {
            MatrixNN net;
            bool exN = Networks.TryGetValue(netN, out net);
            CasesSet cases;
            bool exC = CasesSets.TryGetValue(casesN, out cases);
            if (!exN) {
                Console.WriteLine("Network with that name does not exists.");
                return -1;
            }
            if (!exC) {
                Console.WriteLine("Cases set with that name does not exists.");
                return -1;
            }
            return net.TestNet(cases);
        }
        public void CreateNewCasesSetUniform(string name, int numX, double minX, double maxX, int numY, double minY, double maxY) {
            if (CasesSets.ContainsKey(name))
                Console.WriteLine("Cases set with that name already exists.");
            else
                CasesSets.Add(name, FunctionPart.PrepareCasesUniform(numX, minX, maxX, numY, minY, maxY));
        }
        public void CreateNewCasesSetRandom(string name, int num, double minX, double maxX, double minY, double maxY) {
            if (CasesSets.ContainsKey(name))
                Console.WriteLine("Cases set with that name already exists.");
            else
                CasesSets.Add(name, FunctionPart.PrepareCasesRandom(num, minX, maxX, minY, maxY));
        }
        public void ImportCasesSet(string name, string path) {
            if (CasesSets.ContainsKey(name))
                Console.WriteLine("Cases set with that name already exists.");
            else
                CasesSets.Add(name, ImportExport.ImportCasesSet(path));
        }
        public void ExportCasesSet(string name, string path) {
            CasesSet cs;
            bool ex = CasesSets.TryGetValue(name, out cs);
            if (CasesSets.ContainsKey(name))
                ImportExport.ExportCasesSet(cs, path);
            else
                Console.WriteLine("Cases set with that name does not exists.");
        }
        public void RemoveCasesSet(string name) {
            if (CasesSets.ContainsKey(name))
                CasesSets.Remove(name);
            else
                Console.WriteLine("Cases set with that name does not exists.");
        }
    }

    public class TrainTask
    {
        public Task CurrentTask {get; }
        MatrixNN Network {get; }
        public int Era {get => Network.CurrentEra; }
        public int Element {get => Network.CurrentElement; }
        public int EraCount {get => Network.CountEra; }
        public int ElementCount {get => Network.CountElement; }
        public TrainTask(Task task, MatrixNN net)
        {
            CurrentTask = task;
            Network = net;
        }
    }
}
