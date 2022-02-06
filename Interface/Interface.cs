using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using McMaster.Extensions.CommandLineUtils;
using NeuralNetwork_Console.Models;
using NLog;

namespace NeuralNetwork_Console.Interface
{
    public class Handler
    {
        private Model _model;
        private static readonly NLog.Logger _logger = NLog.LogManager.GetCurrentClassLogger();
        private CommandLineApplication<Handler> processor;

        public Handler()
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            _model = new Model();
            processor = new CommandLineApplication<Handler>();

            processor.Command("exec", (cmd) =>
            {
                cmd.Description = "Execute commands from file";
                var file = cmd.Argument("[file]", "File name").IsRequired();

                cmd.HelpOption("-h|--help");

                cmd.OnExecute(() =>
                {
                    ExecuteFile(file.Value);
                    _logger.Info("File executed");
                    return 0;
                });
            });

            processor.Command("net", (cmd) =>
            {
                cmd.Description = "Do something with nets";

                cmd.HelpOption("-h|--help");

                cmd.Command("new", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Create new feedforward neural network";

                    var nameOption = cmdn.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                    var structureOption = cmdn.Option("-s", "Network structure as \'2;2;3;1\' without \'", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.CreateNewNet(nameOption.Value(), structureOption.Value().Split(";").Select(s => Int32.Parse(s)).ToArray());
                        return 0;
                    });
                });

                cmd.Command("ls", (cmdn) =>
                {
                    cmdn.Description = "Show list of all networks";
                    
                    cmdn.OnExecute(() =>
                    {
                        Console.WriteLine("{0, 15} {1, 40}", "name", "structure");
                        foreach (var n in _model.Networks)
                        {
                            Console.WriteLine("{0, 15} {1, 40}", n.Key, n.Value.Structure);
                        }
                        return 0;
                    });
                });

                cmd.Command("rm", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Remove network by name";

                    var nameOption = cmdn.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.RemoveNet(nameOption.Value());
                        return 0;
                    });
                });

                cmd.Command("import", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Import network from file";

                    var nameOption = cmdn.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                    var fileOption = cmdn.Option("-f|--file", "Path to file", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.ImportNet(nameOption.Value(), fileOption.Value());
                        return 0;
                    });
                });

                cmd.Command("export", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Export network to file";

                    var nameOption = cmdn.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                    var fileOption = cmdn.Option("-f|--file", "Path to file", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.ExportNet(nameOption.Value(), fileOption.Value());
                        return 0;
                    });
                });

                cmd.Command("clone", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Clone network";

                    var originOption = cmdn.Option("-o|--origin", "Original network name", CommandOptionType.SingleValue).IsRequired();
                    var newOption = cmdn.Option("-n|--new", "New network name", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.CopyNet(originOption.Value(), newOption.Value());
                        return 0;
                    });
                });

                cmd.OnExecute(() =>
                {
                    _logger.Info("Please specify command");
                    return 0;
                });
            });

            processor.Command("case", (cmd) =>
            {
                cmd.Description = "Do something with cases";

                cmd.HelpOption("-h|--help");

                cmd.Command("create-test-cases", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Create new cases for network with 2 input and 3 output.\nCases represent truth table for AND, OR and XOR operators";

                    var nameOption = cmdn.Option("-n|--name", "Cases name", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.CreateTestCases(nameOption.Value());
                        _logger.Info("Test cases created");
                        return 0;
                    });
                });

                cmd.Command("ls", (cmdn) =>
                {
                    cmdn.Description = "Show list of all case sets (cases)";

                    cmdn.OnExecute(() =>
                    {
                        Console.WriteLine("{0, 15} {1, 40}", "name", "count");
                        foreach (var n in _model.CasesSets)
                        {
                            Console.WriteLine("{0, 15} {1, 40}", n.Key, n.Value.Count);
                        }
                        return 0;
                    });
                });

                cmd.Command("rm", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Remove case set";

                    var nameOption = cmdn.Option("-n|--name", "Cases name", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.RemoveCasesSet(nameOption.Value());
                        return 0;
                    });
                });

                cmd.Command("import", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Import cases from file";

                    var nameOption = cmdn.Option("-n|--name", "Cases name", CommandOptionType.SingleValue).IsRequired();
                    var fileOption = cmdn.Option("-f|--file", "Path to file", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.ImportCasesSet(nameOption.Value(), fileOption.Value());
                        return 0;
                    });
                });

                cmd.Command("export", (cmdn) =>
                {
                    cmdn.HelpOption("-h|--help");

                    cmdn.Description = "Export Cases to file";

                    var nameOption = cmdn.Option("-n|--name", "Cases name", CommandOptionType.SingleValue).IsRequired();
                    var fileOption = cmdn.Option("-f|--file", "Path to file", CommandOptionType.SingleValue).IsRequired();

                    cmdn.OnExecute(() =>
                    {
                        _model.ExportCasesSet(nameOption.Value(), fileOption.Value());
                        return 0;
                    });
                });

                cmd.OnExecute(() =>
                {
                    _logger.Info("Please specify command");
                    return 0;
                });
            });

            processor.Command("train", (cmd) =>
            {
                cmd.HelpOption("-h|--help");

                cmd.Description = "Train network";

                var nameOption = cmd.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                var caseOption = cmd.Option("-c|--case", "Cases name", CommandOptionType.SingleValue).IsRequired();
                var eraOption = cmd.Option<int>("-e|--era", "Era count (default: 100)", CommandOptionType.SingleValue);
                eraOption.DefaultValue = 100;
                var batchOption = cmd.Option<int>("-b|--batch", "Batch size (default: 100)", CommandOptionType.SingleValue);
                batchOption.DefaultValue = 100;
                var speedOption = cmd.Option<double>("-s|--speed", "Education speed parameter (default: 0.3)", CommandOptionType.SingleValue);
                speedOption.DefaultValue = 0.3;
                var newNameOption = cmd.Option("--new-name", "If specified then trained network will save as new one", CommandOptionType.SingleValue);

                cmd.OnExecute(() =>
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    TrainTask _tt = _model.TrainNet(nameOption.Value(), caseOption.Value(), eraOption.ParsedValue, batchOption.ParsedValue, speedOption.ParsedValue, newNameOption.Value());
                    Timer timer = new Timer(callbackWriteConsole, _tt, 0, 2000);
                    _tt.CurrentTask.Wait();
                    timer.Change(-1, -1);
                    watch.Stop();
                    timer.Dispose();
                    Console.WriteLine("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                    _logger.Info("Trained: {0} seconds", watch.ElapsedMilliseconds / 1000);
                    return 0;
                });
            });

            processor.Command("eval", (cmd) =>
            {
                cmd.HelpOption("-h|--help");

                cmd.Description = "Evaluate network with inputs";

                var nameOption = cmd.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                var inputsOption = cmd.Option("-i|--input", "Input values like \'0.1;2;5\' without \'", CommandOptionType.SingleValue).IsRequired();

                cmd.OnExecute(() =>
                {
                    double[] input = inputsOption.Value().Split(';')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double[] output = _model.EvalValue(nameOption.Value(), input);
                    Console.WriteLine("Evaluated. Result : {0}", output.ToStr());
                    return 0;
                });
            });

            processor.Command("eval-cases", (cmd) =>
            {
                cmd.HelpOption("-h|--help");

                cmd.Description = "Evaluate network with inputs";

                var nameOption = cmd.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                var inputsOption = cmd.Option("-c|--cases", "Cases name", CommandOptionType.SingleValue).IsRequired();
                var resultOption = cmd.Option("-r|--result", "File name for results", CommandOptionType.SingleValue).IsRequired();

                cmd.OnExecute(() =>
                {
                    _model.EvalToFileNet(nameOption.Value(), inputsOption.Value(), resultOption.Value());
                    Console.WriteLine("Evaluated");
                    return 0;
                });
            });

            processor.Command("testcase", (cmd) =>
            {
                cmd.HelpOption("-h|--help");

                cmd.Description = "Test network with one test case";

                var nameOption = cmd.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                var inputsOption = cmd.Option("-i|--input", "Input values like \'0.1;2;5\' without \'", CommandOptionType.SingleValue).IsRequired();
                var outputsOption = cmd.Option("-o|--output", "Output values like \'0.1;2;5\' without \'", CommandOptionType.SingleValue).IsRequired();

                cmd.OnExecute(() =>
                {
                    double[] input = inputsOption.Value().Split(';')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double[] output = outputsOption.Value().Split(';')
                        .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                    double err = _model.TestNet(nameOption.Value(), input, output);
                    Console.WriteLine("Tested. Error : {0}", err);
                    return 0;
                });
            });

            processor.Command("test", (cmd) =>
            {
                cmd.HelpOption("-h|--help");

                cmd.Description = "Test network with cases";

                var nameOption = cmd.Option("-n|--name", "Network name", CommandOptionType.SingleValue).IsRequired();
                var casesOption = cmd.Option("-c|--cases", "Cases name", CommandOptionType.SingleValue).IsRequired();

                cmd.OnExecute(() =>
                {
                    double err = _model.TestNet(nameOption.Value(), casesOption.Value());
                    Console.WriteLine("Tested. Error : {0}", err);
                    return 0;
                });
            });

            processor.Command("exit", (cmd) =>
            {
                cmd.Description = "Exit app";
                
                cmd.OnExecute(() =>
                {
                    _logger.Info("Exit CLI");
                    return -1;
                });
            });

            processor.Command("help", (cmd) =>
            {
                cmd.Description = "Print help message";
                
                cmd.OnExecute(() =>
                {
                    processor.ShowHelp();
                });
            });
        }

        public int ProcessCommand(string command)
        {
            _logger.Info(">> {0}", command);

            try
            {
                int a = processor.Execute(command.Split(' '));

                return a;
            }
            catch (UnrecognizedCommandParsingException e)
            {
                _logger.Error(e);
                Console.WriteLine("Unknown command");
                return 0;
            }
        }

        private void ExecuteFile(string path)
        {
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string? line;

            while (!sr.EndOfStream)
            {
                line = sr.ReadLine();
                Console.WriteLine("> " + line);
                _logger.Info("> {0}", line);
                ProcessCommand(line);
            }

            sr.Close();
            fs.Close();
        }

        private static void callbackWriteConsole(object tt)
        {
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            Console.WriteLine("");
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            int len1 = ((TrainTask)tt).EraCount.ToString().Length;
            string l1 = "{0, " + len1 + "}";
            int len2 = ((TrainTask)tt).EraCount.ToString().Length;
            string l2 = "{0, " + len2 + "}";
            Console.WriteLine("Era {0} / {1} - Element {2} / {3}", String.Format(l1, ((TrainTask)tt).Era), ((TrainTask)tt).EraCount, String.Format(l2, ((TrainTask)tt).Element), ((TrainTask)tt).ElementCount);
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
