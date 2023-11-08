from pyspark.sql import SparkSession


def get_spark():
    # Configure Spark.
    spark = (
        SparkSession.builder.config(
            "spark.jars.packages",
            (
                "uk.co.gresearch.spark:spark-extension_2.12:2.8.0-3.4,"
                "saurfang:spark-sas7bdat:3.0.0-s_2.12"
            ),
        )
        .config("spark.sql.optimizer.maxIterations", "1000")
        .config(
            "spark.driver.memory",
            "64G",
        )
        .master("local[*]")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    # Import the diffing library which empowers the `spark` instance.
    import gresearch.spark.diff  # noqa: F401

    return spark
