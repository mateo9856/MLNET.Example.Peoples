using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNET.Example.Peoples
{
    public class PeopleModel
    {
        [LoadColumn(2), ColumnName("First Name")]
        public string FirstName { get; set; }
        [LoadColumn(3), ColumnName("Last Name")]
        public string LastName { get; set; }
        [LoadColumn(4), ColumnName("Sex")]
        public string Sex { get; set; }
        [LoadColumn(8), ColumnName("Job Title")]
        public string JobTitle { get; set; }
    }

    public class PeopleModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Sex { get; set; }
    }

}
