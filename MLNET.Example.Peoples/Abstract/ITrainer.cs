using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples.Abstract
{
    public interface ITrainer
    {
        void Fit(string filePath);
        void SaveToFile(string file);
        BinaryClassificationMetrics EvaluateBinaryClassification();
        MulticlassClassificationMetrics EvaluateMulticlassClassification();
        RegressionMetrics EvaluateRegression();
        

    }
}
