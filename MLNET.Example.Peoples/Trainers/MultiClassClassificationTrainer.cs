using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples.Trainers
{
    public class MultiClassClassificationTrainer : TrainerServiceBase
    {

        protected override IEstimator<ITransformer> PreparePipeline()
        {
            throw new NotImplementedException();
        }
    }
}
