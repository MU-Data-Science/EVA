/**
 * University of Missouri-Columbia
 * 2020
 */

import sys.process._
import scala.sys.process
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import org.apache.log4j.Logger
import java.util.Calendar

import org.apache.spark.HashPartitioner

// Futures code is taken from http://www.russellspitzer.com/2017/02/27/Concurrency-In-Spark/
object ConcurrentContext {
  import scala.util._
  import scala.concurrent._
  import scala.concurrent.ExecutionContext.Implicits.global
  import scala.concurrent.duration.Duration
  import scala.concurrent.duration.Duration._
  /** Wraps a code block in a Future and returns the future */
  def executeAsync[T](f: => T): Future[T] = {
    Future(f)
  }

  /** Awaits only a set of elements at a time. At most batchSize futures will ever
   * be in memory at a time*/
  def awaitBatch[T](it: Iterator[Future[T]], batchSize: Int = 1, timeout: Duration = Inf) = {
    it.grouped(batchSize)
      .map(batch => Future.sequence(batch))
      .flatMap(futureBatch => Await.result(futureBatch, timeout))
  }

  def awaitSliding[T](it: Iterator[Future[T]], batchSize: Int = 3, timeout: Duration = Inf): Iterator[T] = {
    val slidingIterator = it.sliding(batchSize - 1).withPartial(true) //Our look ahead (hasNext) will auto start the nth future in the batch
    val (initIterator, tailIterator) = slidingIterator.span(_ => slidingIterator.hasNext)
    initIterator.map( futureBatch => Await.result(futureBatch.head, timeout)) ++
      tailIterator.flatMap( lastBatch => Await.result(Future.sequence(lastBatch), timeout))
  }
}

object GenerateDenovoReference {
  def usage(): Unit = {
    println("""
    Usage: spark-submit [Spark options] eva_some_version.jar [jar options]

    Spark options: --master, --num-executors, etc.

    Required jar options:
      -i | --input <file>     input HDFS file containing sample IDs; one per line
      -o | --output <file>    output HDFS directory to store the denovo reference sequences
    """)
  }


  def runSpade[T](x: T):T = {
    println(s"Starting Spades on ($x)")
    //Thread.sleep(15000)
    //val now = Calendar.getInstance()
    val sampleID = x.toString
    val cleanUp = "rm -rf /mydata/$sampleID-*"
    val cleanRet = Process(cleanUp).!
    val cmd =
      s"/proj/eva-public-PG0/SPAdes-3.14.1-Linux/bin/spades.py -m 54 -t 16 --tmp-dir /mydata/$sampleID-tmp" +
      s" -1 /proj/eva-public-PG0/denovo/$sampleID" + "_1.filt.fastq.gz" +
      s" -2 /proj/eva-public-PG0/denovo/$sampleID" + "_2.filt.fastq.gz" +
      s" -o /mydata/$sampleID-output"

    val ret = Process(cmd).!
    //val ret = System.getenv("HOSTNAME")
    println(s"Completed ($x); $cmd; execution status: $ret")
    x
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 1) {
      usage()
      sys.exit(2)
    }

    val argList = args.toList
    type OptionMap = Map[Symbol, Any]

    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case ("-h" | "--help") :: tail => usage(); sys.exit(0)
        case ("-i" | "--input") :: value :: tail => nextOption(map ++ Map('input -> value), tail)
        case ("-o" | "--output") :: value :: tail => nextOption(map ++ Map('output -> value), tail)
        case value :: tail => println("Unknown option: "+value)
          usage()
          sys.exit(1)
      }
    }

    val options = nextOption(Map(),argList)

    val spark = SparkSession.builder.appName("De novo sequence generation").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    val log = Logger.getLogger(getClass.getName)
    log.info("\uD83D\uDC49 Starting the generation")

    val fileName = options('input).toString
    val sequenceList = spark.sparkContext.textFile(fileName)
    val pairList = sequenceList.map(x => (x,1)).partitionBy(
      new HashPartitioner(sequenceList.count().toInt))

    pairList
      .map(x => ConcurrentContext.executeAsync(runSpade(x._1)))
      .mapPartitions(it => ConcurrentContext.awaitBatch(it))
      .collect()
      .foreach(x => println(s"Finishing with $x on hostname"))

    log.info("\uD83D\uDC49 Completed the generation")
    spark.stop()
  }
}