using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using NeuralNetwork_Console.Models;
using NLog;

namespace NeuralNetwork_Console.Interface {
    public class MainInterface {
        private MathModel _model;

        private static readonly NLog.Logger Logger = NLog.LogManager.GetCurrentClassLogger();

        public MainInterface() {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            Logger.Info("Start");
            _model = new MathModel();
            MainLoop();
            Console.WriteLine("Interface exit");
            Logger.Info("Exit");

        }

        private void MainLoop() {
            Console.WriteLine("Process started");
            int result = 1;
            while (result > -1) {
                Console.Write(">> ");
                Task<string> w = Task.Run(() => Console.ReadLine());
                w.Wait();
                string command = w.Result;
                Logger.Info(">> {0}", command);
                command = command.ToLower();
                result = ProcessCommand(command);
            }
            return;
        }

        private int ProcessCommand(string command) {
            try {
                // string[] parts = command.Split(' ');
                string[] parts = Regex.Matches(command, @"(\S+)").Select(e => e.Groups[1].Value).ToArray();
                switch (parts[0]) {
                case "exit":
                    return -1;
                case "exec":
                    ExecuteFile(parts[1]);
                    Console.WriteLine("File executed");
                    Logger.Info("File executed");
                    break;
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
                    case "copy":
                        _model.CopyNet(parts[2], parts[3]);
                        break;
                    }
                    break;
                case "case":
                    switch (parts[1]) {
                    case "new":
                        switch (parts[2]) {
                        case "uniform":
                            _model.CreateNewCasesSetUniform(parts[3], Int32.Parse(parts[4]), Double.Parse(parts[5], CultureInfo.InvariantCulture), Double.Parse(parts[6], CultureInfo.InvariantCulture), Int32.Parse(parts[7]), Double.Parse(parts[8]), Double.Parse(parts[9], CultureInfo.InvariantCulture));
                            break;
                        case "random":
                            _model.CreateNewCasesSetRandom(parts[3], Int32.Parse(parts[4]), Double.Parse(parts[5], CultureInfo.InvariantCulture), Double.Parse(parts[6], CultureInfo.InvariantCulture), Double.Parse(parts[7], CultureInfo.InvariantCulture), Double.Parse(parts[8], CultureInfo.InvariantCulture));
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
                        Timer timer = new Timer(callbackWriteConsole, _tt, 0, 2000);
                        Task.Delay(20);
                        _tt.CurrentTask.Wait();
                        timer.Change(-1, -1);
                        watch.Stop();
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        Logger.Info("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    if (parts.Length == 4) {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        // net, case
                        TrainTask _tt = _model.TrainNet(parts[1], parts[2], resNetName : parts[3]);
                        Timer timer = new Timer(callbackWriteConsole, _tt, 1000, 2000);
                        _tt.CurrentTask.Wait();
                        timer.Change(-1, -1);
                        watch.Stop();
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        Logger.Info("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    if (parts.Length == 6) {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        // net, case, epochs, batch, speed
                        TrainTask _tt = _model.TrainNet(parts[1], parts[2], Int32.Parse(parts[3]), Int32.Parse(parts[4]), Double.Parse(parts[5], CultureInfo.InvariantCulture));
                        Timer timer = new Timer(callbackWriteConsole, _tt, 1000, 2000);
                        _tt.CurrentTask.Wait();
                        timer.Change(-1, -1);
                        watch.Stop();
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        Logger.Info("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    if (parts.Length == 7) {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        // net, case, edu, batch, speed
                        TrainTask _tt = _model.TrainNet(parts[1], parts[2], Int32.Parse(parts[4]), Int32.Parse(parts[5]), Double.Parse(parts[6], CultureInfo.InvariantCulture), resNetName : parts[3]);
                        Timer timer = new Timer(callbackWriteConsole, _tt, 1000, 2000);
                        _tt.CurrentTask.Wait();
                        timer.Change(-1, -1);
                        watch.Stop();
                        timer.Dispose();
                        Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        Logger.Info("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                        return 0;
                    }
                    break;
                case "eval":
                    Match m = Regex.Match(command, @"{((?:[\d\,\.\-]+\s?)+)}");
                    double[] input1 = m.Groups[1].Value.Split(' ')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double[] out1 = _model.EvalValue(parts[1], input1);
                    Console.WriteLine("Evaluated. Result : {0}", out1.ToStr());
                    Logger.Info("Evaluated. Result : {0}", out1.ToStr());
                    break;
                case "testcase":
                    Match m1 = Regex.Match(command, @"{((?:[\d\,\.\-]+\s?)+)} => {((?:[\d\,\.\-]+\s?)+)}");
                    double[] input = m1.Groups[1].Value.Split(' ')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double[] output = m1.Groups[2].Value.Split(' ')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double err1 = _model.TestNet(parts[1], input, output);
                    Console.WriteLine("Tested. Error : {0}", err1);
                    Logger.Info("Tested. Error : {0}", err1);
                    break;
                case "test":
                    double err = _model.TestNet(parts[1], parts[2]);
                    Console.WriteLine("Tested. Error : {0}", err);
                    Logger.Info("Tested. Error : {0}", err);
                    break;
                case "graphnet":
                    _model.GraphNet(parts[1], parts[2], parts[3]);
                break;
                }
            } catch (FileNotFoundException e) {
                Console.WriteLine("File not found: \"{0}\"", command);
                Logger.Error(e, "File not found");
            } catch (Exception e) {
                Console.WriteLine("Some error in: \"{0}\"", command);
                Logger.Error(e, "Some error");
            }
            return 0;
        }

        private void ExecuteFile(string path) {
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string? line;

            while (!sr.EndOfStream) {
                line = sr.ReadLine();
                Console.WriteLine("> " + line);
                Logger.Info("> {0}", line);
                ProcessCommand(line);
            }

            sr.Close();
            fs.Close();
        }

        private static void callbackWriteConsole(object tt) {
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            Console.WriteLine("");
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            Console.WriteLine("Era {0} / {1} - Element {2} / {3}", ((TrainTask)tt).Era, ((TrainTask)tt).EraCount, ((TrainTask)tt).Element, ((TrainTask)tt).ElementCount);
        }
    }

    public static class Ext
    {
        public static string ToStr(this double[] arr)
        {
            StringBuilder res = new StringBuilder("{");
            foreach (double v in arr)
            {
                res.Append(v);
                res.Append(" ");
            }
            res.Remove(res.Length - 1, 1);
            res.Append("}");
            return res.ToString();
        }

    }

}
