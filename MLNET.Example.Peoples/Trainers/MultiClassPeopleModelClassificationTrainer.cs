using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples.Trainers
{
    public class MultiClassPeopleModelClassificationTrainer : TrainerServiceBase
    {

        public MultiClassPeopleModelClassificationTrainer() : base()
        {

        }
        public EstimatorChain<KeyToValueMappingTransformer> PrepareSdcaMaximumEntropy(IEstimator<ITransformer> dataPipeline, string keyValue)
        {
            return dataPipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue(keyValue));
        }
        protected override IEstimator<ITransformer> PreparePipeline()
        {
            throw new NotImplementedException();
        }
    }
}
