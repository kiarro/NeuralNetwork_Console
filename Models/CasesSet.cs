using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixNeuralNetwok
{
    public class CasesSet : IEnumerable<KeyValuePair<double[], double[]>>
    {
        Dictionary<double[], double[]> cases;
        public int Count { get => cases.Count; }
        public CasesSet()
        {
            this.cases = new Dictionary<double[], double[]>();
        }
        public CasesSet(Dictionary<double[], double[]> cases)
        {
            this.cases = cases;
        }
        public KeyValuePair<double[], double[]> ElementAt(int position)
        {
            return cases.ElementAt(position);
        }

        public IEnumerator<KeyValuePair<double[], double[]>> GetEnumerator()
        {
            return cases.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return cases.GetEnumerator();
        }
        public void Add(double[] key, double[] value)
        {
            cases.Add(key, value);
        }
        public void Remove(double[] key)
        {
            cases.Remove(key);
        }
    }
}
