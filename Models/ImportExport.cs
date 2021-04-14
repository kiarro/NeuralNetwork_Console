using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace MatrixNeuralNetwok.FileStorage
{
    class ImportExport
    {
        public static void ExportNN(MatrixNN network, string path)
        {
            StringBuilder exportString = new StringBuilder();
            exportString.AppendFormat("layers: {0}\n", network.LayerAmount);
            exportString.AppendFormat("str: {0}", network.Structure);
            for (int i = 0; i < network.LayerAmount-1; i++)
            {
                exportString.AppendFormat("w{0}\n", i + 1);
                exportString.Append(network.Weights[i].ToMatrixString(network.Weights[i].RowCount, network.Weights[i].ColumnCount, "N10"));
                exportString.AppendFormat("s{0}\n", i + 1);
                exportString.Append(network.Shift[i].ToVectorString(network.Shift[i].Count, 16, "N10"));
            }

            FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            StreamWriter sw = new StreamWriter(fs);
            sw.WriteLine(exportString);
            sw.Flush();
            sw.Close();
            fs.Close();

            PrintEpochsError(network.errs, path+".EpochStats");
        }

        public static MatrixNN ImportNN(string path)
        {
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string? line;
            // get layer amount line
            sr.ReadLine();
            // get structure
            line = sr.ReadLine();
            int[] str = Regex.Matches(line, @"(\d+)").Cast<Match>().Select(s => Int32.Parse(s.Value)).ToArray();
            MatrixNN network = new MatrixNN(str);
            // get weights and shifts
            for (int i = 0; i < str.Length - 1; i++)
            {
                double[][] w = new double[str[i + 1]][];
                double[] s = new double[str[i + 1]];
                // read line "wi"
                line = sr.ReadLine();
                // read matrix of weights
                for (int j = 0; j < str[i + 1]; j++)
                {
                    line = sr.ReadLine();
                    w[j] = Regex.Matches(line, @"\s*([\d\,\.-]+)")
                        .Cast<Match>()
                        .Select(s => Double.Parse(s.Value, CultureInfo.InvariantCulture))
                        .ToArray();
                }
                network.Weights[i] = Matrix<double>.Build.DenseOfRowArrays(w);
                // read line "si"
                line = sr.ReadLine();
                // read vector of shifts
                for (int j = 0; j < str[i + 1]; j++)
                {
                    line = sr.ReadLine();
                    s[j] = Double.Parse(line, CultureInfo.InvariantCulture);
                }
                network.Shift[i] = Vector<double>.Build.DenseOfArray(s);
            }

            sr.Close();
            fs.Close();
            return network;
        }
        public static void ExportCasesSet(CasesSet set, string path)
        {
            StringBuilder exportString = new StringBuilder();
            exportString.AppendFormat("cases: {0}\n", set.Count);
            foreach (var s in set)
            {
                exportString.Append(s.Key.ToStr());
                exportString.Append(" => ");
                exportString.Append(s.Value.ToStr());
                exportString.Append("\n");
            }

            FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            StreamWriter sw = new StreamWriter(fs);
            sw.WriteLine(exportString);
            sw.Flush();
            sw.Close();
            fs.Close();
        }
        public static CasesSet ImportCasesSet(string path)
        {
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string? line;
            // get set size
            line = sr.ReadLine();
            int size = Int32.Parse(line.Substring(line.LastIndexOf(' ') + 1));
            CasesSet set = new CasesSet();
            // get cases
            for (int i = 0; i < size; i++)
            {
                // read line of case
                line = sr.ReadLine();
                double[] key; double[] value;
                Match m = Regex.Match(line, @"{((?:[\d\,\.\-]+\s?)+)} => {((?:[\d\,\.\-]+\s?)+)}");
                key = m.Groups[1].Value.Split(' ')
                    .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                    .ToArray();
                value = m.Groups[2].Value.Split(' ')
                    .Select(s => Double.Parse(s, CultureInfo.InvariantCulture))
                    .ToArray();
                set.Add(key, value);
            }

            sr.Close();
            fs.Close();
            return set;
        }

        public static void PrintEpochsError(double[] values, string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write);
            StreamWriter sw = new StreamWriter(fs);
            
            sw.WriteLine("{0,10}\t{1,10}", "Epoch", "Error");
            for (int i=0; i<values.Length; i++){
                sw.WriteLine("{0,10}\t{1,10:N10}", i+1, values[i]);
            }
            
            sw.Flush();
            sw.Close();
            fs.Close();
            
        }
    }

    public static class Ext
    {
        public static string ToStr(this double[] arr)
        {
            StringBuilder res = new StringBuilder("{");
            foreach (double v in arr)
            {
                res.Append(v.ToString("N10"));
                res.Append(" ");
            }
            res.Remove(res.Length - 1, 1);
            res.Append("}");
            return res.ToString();
        }

    }
}
