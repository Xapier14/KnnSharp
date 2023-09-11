using System.Collections;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;

namespace Xapier14.KnnSharp
{
    public class DataSet<T> : IEnumerable<DataValue<T>[]> where T : INumber<T>
    {
        private readonly List<DataPoint<T>> _list = new();
        private readonly int _fields;
        private string[] _labels;

        public int FieldCount => _fields;

        public DataSet(int fieldCount)
        {
            _fields = fieldCount;
            _labels = new string[fieldCount];
        }

        public void SetDataLabels(params string[] columnLabels)
        {
            var i = 0;
            foreach (var columnLabel in columnLabels)
            {
                if (i >= _fields)
                    break;
                _labels[i] = columnLabel;
                i++;
            }
        }

        public string? GetDataLabel(int columnIndex)
            => _labels[columnIndex];

        public void AddPoint(params DataValue<T>[] values)
        {
            if (values.Length > _fields)
                throw new ArgumentOutOfRangeException(nameof(values), "Number of parameters exceeds field count.");
            var dataPoint = new DataPoint<T>(_fields);
            for (var i = 0; i < values.Length; ++i)
            {
                dataPoint[i] = values[i];
            }
            _list.Add(dataPoint);
        }

        public void AddPoint(DataPoint<T> dataPoint)
        {
            _list.Add(dataPoint);
        }

        public DataValue<T>[,] ToMatrix()
        {
            var n = _list.Count;
            var matrix = new DataValue<T>[n, _fields];
            for (var i = 0; i < n; ++i)
            {
                for (var v = 0; v < _fields; ++v)
                {
                    matrix[i, v] = _list[i][v];
                }
            }

            return matrix;
        }

        public DataValue<T>[][] ToArray()
        {
            var n = _list.Count;
            // TODO: test
            var arr = new DataValue<T>[n][];
            for (var i = 0; i < n; ++i)
            {
                for (var v = 0; v < _fields; ++v)
                {
                    arr[i][v] = _list[i][v];
                }
            }

            return arr;
        }

        public DataValue<T>[] GetColumn(int index)
        {
            var n = _list.Count;
            if (index < 0 || index >= _fields)
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");
            var arr = new DataValue<T>[n];
            for (var i = 0; i < n; ++i)
                arr[i] = _list[i][index];

            return arr;
        }

        public DataValue<T>[] GetRow(int index)
        {
            var n = _list.Count;
            if (index < 0 || index >= n)
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");
            var arr = new DataValue<T>[_fields];
            for (var i = 0; i < _fields; ++i)
                arr[i] = _list[index][i];

            return arr;
        }

        public DataSet<T> RemoveColumn(int index)
        {
            var dataSet = new DataSet<T>(FieldCount - 1);
            foreach (var oldPoint in _list)
            {
                var newPoint = new DataPoint<T>(FieldCount - 1);
                var i = 0;
                for (var j = 0; j < oldPoint.Length; ++j)
                {
                    if (j == index)
                        continue;
                    newPoint[i] = oldPoint[j];
                    i++;
                }
                dataSet.AddPoint(newPoint);
            }

            return dataSet;
        }

        public IEnumerator<DataValue<T>[]> GetEnumerator()
        {
            return _list.Select(entry => entry.ToArray()).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public static DataSet<T> LoadFromCsvFile(string csvPath, IFormatProvider? formatProvider = null)
        {
            var maxFields = 0;
            var values = new List<string[]>();
            using var reader = File.OpenText(csvPath);
            var line = reader.ReadLine();
            while (line != null)
            {
                var split = line.Split(',', StringSplitOptions.TrimEntries);
                if (split.Length > maxFields)
                    maxFields = split.Length;
                if (line != string.Empty)
                    values.Add(split);
                line = reader.ReadLine();
            }

            var dataSet = new DataSet<T>(maxFields);
            foreach (var valueLine in values)
            {
                var dataPoint = new DataPoint<T>(maxFields);
                for (var i = 0; i < valueLine.Length; ++i)
                {
                    dataPoint[i] = T.TryParse(valueLine[i], formatProvider ?? new NumberFormatInfo(), out var value)
                        ? new DataValue<T>(value)
                        : new DataValue<T>(valueLine[i]);
                }

                dataSet.AddPoint(dataPoint);
            }

            return dataSet;
        }
    }
}
