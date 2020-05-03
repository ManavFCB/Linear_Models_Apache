package tricount
import org.apache.log4j.LogManager
import org.apache.spark.{SparkConf, SparkContext}
import scala.io.Source
import scala.collection.mutable
import scala.collection.mutable.{LinkedList}
object LinearReg {
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nLinearReg main <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val tol=0.00001
    //var pgd_map=new mutable.HashMap[Integer,Double]()
    val weights = sc.textFile(args(1)).map(line=>{
      val s=line.split(",")
      (s(0).toInt,s(1).toDouble)
    }).collect
    var curr_error = Double.PositiveInfinity;
    var flag=0
    while(flag==0){
    val weights_br = sc.broadcast(weights)
    var upd=new LinkedList[(Integer,Double)]()
    val new_error = sc.doubleAccumulator("Current Mean Squared Error")
    val n = sc.longAccumulator("no of observations")
    val inp_rdd = sc.textFile(args(0))
    val pgd = inp_rdd.map(line => {
      val s = line.split(",")
      val wts = weights_br.value
      var sum = wts(0)._2
      for (i <- 1 to wts.size - 1) {
        sum = sum + wts(i)._2 * s(i).toDouble
      }
      val diff = sum - s(0).toDouble
      var pgd_map=new mutable.HashMap[Integer,Double]()
      new_error.add(math.pow(diff, 2))
      n.add(1)
      for (i <- 0 to wts.size - 1) {
        if (i == 0) {
          pgd_map.put(i,diff)
        } else {
            pgd_map.put(i,diff*s(i).toDouble)
        }
      }
     pgd_map
    }).collect()
    val mse=new_error.value/n.value
    val nob=n.value
    if(math.abs(mse-curr_error)>tol && mse<curr_error){
      var arr=new Array[(Integer,Double)](pgd.length*weights.size)
      var k=0
      for(i<- 0 to pgd.length-1){
        for(t<-0 to pgd(i).keySet.size-1){
          arr(k)=(t,pgd(i)(t))
          k=k+1
        }
      }
      var new_wts=sc.parallelize(arr.toSeq).reduceByKey(_+_).map(line=>{
        val wts=weights_br.value(line._1)
        (line._1,wts._2-0.00001*2.0*line._2/nob)
      }).collect()
      for(i<-0 to new_wts.length-1){
        weights(new_wts(i)._1)=(new_wts(i)._1,new_wts(i)._2)
      }
      curr_error=mse
    }
      else{
      val weights_rdd=sc.parallelize(weights)
      weights_rdd.saveAsTextFile("s3://mr-manav-bucket-1/output")
      flag=1
    }
    }
  }
}
