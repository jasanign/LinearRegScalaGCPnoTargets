// Databricks notebook source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{ StructType, StructField, StringType, IntegerType, DoubleType }
import java.io.File
import java.io.PrintWriter
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level

// Application and context information

// COMMAND ----------
object linear2 {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    // Application and context information
    val config = new SparkConf().setAppName("LinearRegression").setMaster("local")
    val writer = new PrintWriter(new File("r2_output.txt"))
    //val sc = new SparkContext(config)

    // COMMAND ----------

    val schema = StructType(
      StructField("ID", IntegerType, false) ::
        StructField("year", IntegerType, false) ::
        StructField("month", IntegerType, false) ::
        //StructField("month", StringType, false) ::
        StructField("ngas_ft3", IntegerType, true) ::
        StructField("ngas_cost", DoubleType, true) ::
        StructField("pgas_ft3", IntegerType, true) ::
        //StructField("pgas_cost", DoubleType, true) ::
        StructField("oil_gal", IntegerType, true) ::
        //StructField("oil_cost", DoubleType, true) ::
        StructField("makeup_wat_gal", IntegerType, true) ::
        StructField("makeup_wat_cost", DoubleType, true) ::
        StructField("avg_feedback_temp", IntegerType, true) ::
        StructField("electric_kwh", IntegerType, true) ::
        StructField("elec_cost", DoubleType, true) ::
        StructField("elec_cost_per_kwh", DoubleType, true) ::
        StructField("gen_steam_lb", IntegerType, true) ::
        StructField("gen_steam_no1_lb", IntegerType, true) ::
        StructField("gen_steam_no2_lb", IntegerType, true) ::
        StructField("gen_steam_no3_lb", IntegerType, true) ::
        StructField("gen_steam_no4_lb", IntegerType, true) ::
        StructField("boiler_efficiency", IntegerType, true) ::
        StructField("steam_cost_fuel_per_mlb", DoubleType, true) ::
        StructField("steam_cost_total_per_mlb", DoubleType, true) ::
        //StructField("hwater_prod_gal", DoubleType, true) ::
        //StructField("steam_for_hwater_lb", DoubleType, true) :: 
        //StructField("steam_for_heating_lb", DoubleType, true) ::
        StructField("hdd", IntegerType, true) ::
        StructField("cdd", IntegerType, true) ::
        StructField("mean_outside_temp", DoubleType, true) ::
        StructField("fyear", StringType, true) ::
        StructField("ngas_per_dt", DoubleType, true) ::
        StructField("dt", IntegerType, true) :: Nil
    )

    //    val table = sqlContext.read
    //      .format("com.databricks.spark.csv")
    //      .option("header", "true")
    //      .schema(schema)
    //      .load("dbfs:/FileStore/tables/foif3ihy1479226801699/pivot.csv")
    //      //.load("dbfs:/FileStore/tables/fj3gzn1t1486146062744/test_pivot-f1153.csv")
    //    ///FileStore/tables/kvnmw58l1486148959099/test_pivot-f1153.csv
    //      .cache()
    //    
    //    // COMMAND ----------

    //    val session = org.apache.spark.sql.SparkSession.builder
    //      .master("local")
    //      .appName("Spark CSV Reader")
    //      .config("spark.sql.warehouse.dir", "file:///L:/WORKSPACES/cs496_workspace/LinearRegression2/data/pivot_no_outliers.csv")
    //      .getOrCreate;

    val session = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .config("spark.sql.warehouse.dir", "pivot_no_outliers.csv")
      .getOrCreate;

    val table = session.read
      .format("org.apache.spark.csv")
      .option("header", "true") //reading the headers
      .schema(schema)
      .csv("gs://dataproc-15a7be6e-fb43-4cbd-a316-942b0e1241c1-us/pivot_no_outliers.csv")
      .cache()
    table.show()
    //table.select("month").show()
    //val monthTable = table.select("*").where("month = 2")
    //monthTable.show();
    // COMMAND ----------

    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.regression.LinearRegression
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
    import org.apache.spark.sql.functions.udf

    /**
     * Here we build the model of our data and separate our data into features and labels.
     * Our linear regression model is fit to our data.
     */
    //    var month  = 0;
    //    for( month <- 1 to 300){
    val makeFV = udf { (a: Int /*, b: Int*/ ) => Vectors.dense(a /*, b*/ ) } // Use only the fields we plan on using in our regression model
    val doLabel = udf { (a: Int) => a.toDouble }

    val labelledData = table.withColumn("features", makeFV(table("hdd"))).withColumn("label", doLabel(table("dt"))).select( /*"year", "month",*/ "features", "label")
    val Array(training, testing) = labelledData.randomSplit(Array(0.9, 0.1))

    //        print("Training table: \n")
    //        training.show()
    //        print("Testing table: \n")
    //        testing.show()

    val lireg = new LinearRegression()
    //.setMaxIter(10)
    //.setStandardization(true)

    /**
     * CROSS-VALIDATION
     *
     * Inspired by documentation done at
     * https://spark.apache.org/docs/latest/ml-tuning.html and
     * http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.regression.DecisionTreeRegressor.
     */

    // DT Params: maxBins, checkpointInterval, maxDepth, minInfoGain, minInstancesPerNode, seed
    val pgrid = new ParamGridBuilder()
      .addGrid(lireg.maxIter, Array(1, 5, 10))
      .build()

    val cv = new CrossValidator()
      .setEstimator(lireg)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(pgrid)
      .setNumFolds(5) // Use 3+ in practice

    val cvmodel = cv.fit(training)
    val cvtrans = cvmodel.transform(testing)
    cvtrans.show()
    // precision, accuracy
    /* More evaluation information at https://spark.apache.org/docs/1.6.1/api/java/org/apache/spark/ml/evaluation/RegressionEvaluator.html */
    val r2 = new RegressionEvaluator().setMetricName("r2").evaluate(cvtrans)

    /*val liregModel = lireg.fit(training)
        val liregPred = liregModel.transform(testing)
        val liregEval = new RegressionEvaluator().setMetricName("r2")
        
        val r2 = liregEval.evaluate(liregPred)
        liregPred.show()*/

    print("r2 = " + r2 + "\n")

    //        print(r2+"\t-> "+month+"\n")
    //        writer.write(month+".\tr2 = " + r2 + "\n")
    //    
    //    }
    //    writer.close()
  }

}