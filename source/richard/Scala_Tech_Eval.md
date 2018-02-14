# Scala Tech Evaluation

## Python Compatability
There are three popular options for combining code from Scala and Python. These three options are pretty much methods for converting Python to Java code.

1. Jython - version of Python that runs on JVM. Will require entire team to write pipeline code in this language and then we can port it into Scala application. Also requires additional layers such as JyNI to run libraries such as Numpy.

2. PySpark - use Spark library which wraps Python code but handles all the large parallel programming applications with Scala. You can use libraries such as NumPy, etc; however there is a lot of overhead and bugs that come from Spark.

3. Jep - Allows you to embed Python code into Scala. You can import Python modules and directly run the code. In reality, you are still just running Python code using a Python interpreter. The GIL still applies and you really gain nothing.


## Scala Concurrency

From what I've read so far, Scala is great for handling big data and serving as an application for launching big data jobs. However, for the actual data science process, Python serves as the better final layer due to its libraries. Scala does have its own set of machine learning/ stats libraries but they just aren't as good.

From the architecture of PySpark, we can however see that Scala might be useful when we arrive at distributed computing. Scala seems to be much faster and better than Python when it comes to generating and controlling workers on different processes running the same pipeline.

That is, we cannot use Scala to parallelize the algorithm but can use it to run multiple copies of the same algorithm across multipile nodes. 

Sources:

https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html

https://www.dezyre.com/article/scala-vs-python-for-apache-spark/213

https://github.com/ninia/jep/wiki/Jep-and-the-GIL

https://github.com/scalala/Scalala/wiki

https://datasciencevademecum.wordpress.com/2016/01/28/6-points-to-compare-python-and-scala-for-data-science-using-apache-spark/

https://cwiki.apache.org/confluence/display/SPARK/PySpark+Internals
