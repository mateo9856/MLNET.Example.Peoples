using Microsoft.ML;
using Microsoft.ML.Data;
using MLNET.Example.Peoples.Abstract;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples
{
    public class TrainerService : ITrainer
    {
        private MLContext _mlContext;
        private ITransformer _trainingModel;
        private IDataView data;
        public TrainerService()
        {
            _mlContext = new MLContext();
        }

        public TrainerService(int seed)
        {
            _mlContext = new MLContext(seed: seed);
        }

        public BinaryClassificationMetrics EvaluateBinaryClassification()
        {
            throw new NotImplementedException();
        }

        public MulticlassClassificationMetrics EvaluateMulticlassClassification()
        {
            throw new NotImplementedException();
        }

        public RegressionMetrics EvaluateRegression()
        {
            throw new NotImplementedException();
        }

        public void Fit(string filePath)
        {
            data = _mlContext.Data.LoadFromTextFile(filePath);
            var processPipeline = PreparePipeline();
            _trainingModel = processPipeline.Fit(data);
        }

        public void SaveToFile(string file)
        {
            if(_trainingModel != null && data != null)
                _mlContext.Model.Save(_trainingModel, data.Schema, file);
        }

        protected virtual IEstimator<ITransformer> PreparePipeline() {
            throw new Exception("You use a base Pipeline, use inherit class!");
        }

    }
}
