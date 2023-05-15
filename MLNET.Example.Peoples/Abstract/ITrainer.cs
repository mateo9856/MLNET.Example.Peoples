using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples.Abstract
{
    public interface ITrainer<T> where T : class
    {
        void Fit(string filePath);
        void SaveToFile();

    }
}
