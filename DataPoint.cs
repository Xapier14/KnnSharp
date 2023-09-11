using System.Collections;
using System.Numerics;

namespace Xapier14.KnnSharp
{
    public class DataPoint<T> : IEnumerable<DataValue<T>> where T : INumber<T>
    {
        private readonly DataValue<T>[] _array;
        public int Length => _array.Length;

        public DataPoint(int length)
        {
            _array = new DataValue<T>[length];
        }
        
        public DataValue<T> this[int index]
        {
            get => _array[index];
            set => _array[index] = value;
        }

        public DataValue<T>[] ToArray()
            => _array;

        public DataPoint<T> ExtractWithoutIndex(int index)
        {
            var dataPoint = new DataPoint<T>(Length - 1);
            var i = 0;
            for (var j = 0; j < Length; ++j)
            {
                if (j == index)
                    continue;
                dataPoint[i] = _array[j];
                i++;
            }

            return dataPoint;
        }

        public IEnumerator<DataValue<T>> GetEnumerator()
        {
            return ((IEnumerable<DataValue<T>>)_array).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public static implicit operator T[](DataPoint<T> dataPoint)
        {
            var arr = new T[dataPoint.Length];
            for (var i = 0; i < dataPoint.Length; ++i)
                arr[i] = dataPoint[i].GetNumber();
            return arr;
        }

        public static implicit operator DataValue<T>[](DataPoint<T> dataPoint)
        {
            var arr = new DataValue<T>[dataPoint.Length];
            for (var i = 0; i < dataPoint.Length; ++i)
                arr[i] = dataPoint[i];
            return arr;
        }

        public static implicit operator DataPoint<T>(DataValue<T>[] dataValues)
        {
            var dataPoint = new DataPoint<T>(dataValues.Length);
            for (var i = 0; i < dataValues.Length; ++i)
                dataPoint[i] = dataValues[i];
            return dataPoint;
        }
    }
}
