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
        [LoadColumn(0), ColumnName("Index")]
        public int Index { get; set; }
        [LoadColumn(1), ColumnName("User Id")]
        public string UserId { get; set; }
        [LoadColumn(2), ColumnName("First Name")]
        public string FirstName { get; set; }
        [LoadColumn(3), ColumnName("Last Name")]
        public string LastName { get; set; }
        [LoadColumn(4), ColumnName("Sex")]
        public string Sex { get; set; }
        [LoadColumn(5), ColumnName("Email")]
        public string Email { get; set; }
        [LoadColumn(6), ColumnName("Phone")]
        public string Phone { get; set; }
        [LoadColumn(7), ColumnName("Date of birth")]
        public string DateOfBirth { get; set; }
        [LoadColumn(8), ColumnName("Job Title")]
        public string JobTitle { get; set; }
    }

    public class PeopleModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Sex { get; set; }
    }

}
