using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using NeuralNetwork_Console.Models;

namespace NeuralNetwork_Console.Interface {
    public class MainInterface {
        private MathModel _model;

        public MainInterface() {
            _model = new MathModel();
            MainLoop();
            Console.WriteLine("Interface exit");
        }

        private void MainLoop() {
            Console.WriteLine("Process started");
            int result = 1;
            while (result > -1) {
                Task<string> w = Task.Run(() => Console.ReadLine());
                w.Wait();
                string command = w.Result;
                command = command.ToLower();
                result = ProcessCommand(command);
            }
            return;
        }

        private int ProcessCommand(string command) {
            try {
                string[] parts = command.Split(' ');
                switch (parts[0]) {
                case "exit":
                    return -1;
                case "net":
                    switch (parts[1]) {
                    case "new":
                        _model.CreateNewNet(parts[2], Regex.Matches(parts[3], @"(\d+)").Cast<Match>().Select(s => Int32.Parse(s.Value)).ToArray());
                        break;
                    case "ls":
                        Console.WriteLine("{0, 15} {1, 40}", "name", "structure");
                        foreach (var n in _model.Networks) {
                            Console.WriteLine("{0, 15} {1, 40}", n.Key, n.Value.Structure);
                        }
                        break;
                    case "rm":
                        _model.RemoveNet(parts[2]);
                        break;
                    case "import":
                        _model.ImportNet(parts[2], parts[3]);
                        break;
                    case "export":
                        _model.ExportNet(parts[2], parts[3]);
                        break;
                    }
                    break;
                case "case":
                    switch (parts[1]) {
                    case "new":
                        switch (parts[2]) {
                        case "uniform":
                            _model.CreateNewCasesSetUniform(parts[3], Int32.Parse(parts[4]), Double.Parse(parts[5]), Double.Parse(parts[6]), Int32.Parse(parts[7]), Double.Parse(parts[8]), Double.Parse(parts[9]));
                            break;
                        case "random":
                            _model.CreateNewCasesSetRandom(parts[3], Int32.Parse(parts[4]), Double.Parse(parts[5]), Double.Parse(parts[6]), Double.Parse(parts[7]), Double.Parse(parts[8]));
                            break;
                        }
                        break;
                    case "ls":
                        Console.WriteLine("{0, 15} {1, 40}", "name", "count");
                        foreach (var n in _model.CasesSets) {
                            Console.WriteLine("{0, 15} {1, 40}", n.Key, n.Value.Count);
                        }
                        break;
                    case "rm":
                        _model.RemoveCasesSet(parts[2]);
                        break;
                    case "import":
                        _model.ImportCasesSet(parts[2], parts[3]);
                        break;
                    case "export":
                        _model.ExportCasesSet(parts[2], parts[3]);
                        break;
                    }
                    break;
                case "train":
                    Console.WriteLine("");
                    if (parts.Length == 3) {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        // net, case
                        TrainTask _tt = _model.TrainNet(parts[1], parts[2]);
                        Timer timer = new Timer(callbackWriteConsole, _tt, 1000, 2000);
                        _tt.CurrentTask.Wait();
                        watch.Stop();
                        timer.Change(-1, -1);
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    if (parts.Length == 6) {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        // net, case, edu, batch, speed
                        TrainTask _tt = _model.TrainNet(parts[1], parts[2], Int32.Parse(parts[3]), Int32.Parse(parts[4]), Double.Parse(parts[5]));
                        Timer timer = new Timer(callbackWriteConsole, _tt, 1000, 2000);
                        _tt.CurrentTask.Wait();
                        watch.Stop();
                        timer.Change(-1, -1);
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    break;
                case "test":
                    double err = _model.TestNet(parts[1], parts[2]);
                    Console.WriteLine("Tested. Error : {0}", err);
                    break;
                }
            } catch (Exception e) {
                Console.WriteLine("Some error");
            }
            return 0;
        }

        private static void callbackWriteConsole(object tt) {
            Console.SetCursorPosition(0, Console.CursorTop-1);
            Console.WriteLine("");
            Console.SetCursorPosition(0, Console.CursorTop-1);
            Console.WriteLine("Era {0} / {1} - Element {2} / {3}", ((TrainTask)tt).Era, ((TrainTask)tt).EraCount, ((TrainTask)tt).Element, ((TrainTask)tt).ElementCount);
        }
    }

}
