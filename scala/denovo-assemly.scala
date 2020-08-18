import sys.process._
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import spark.implicits._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

// http://www.russellspitzer.com/2017/02/27/Concurrency-In-Spark/
object ConcurrentContext {
  import scala.util._
  import scala.concurrent._
  import scala.concurrent.ExecutionContext.Implicits.global
  /** Wraps a code block in a Future and returns the future */
  def executeAsync[T](f: => T): Future[T] = {
    Future(f)
  }
}

var j = 0
val maxConcurrentFutures = 4
var matFutures = ListBuffer[Future[Boolean]]()
var timeout = Duration("Inf")

def slowFoo[T](x: T):T = {
  println(s"slowFoo start ($x)")
  Thread.sleep(5000)
  println(s"slowFoo end($x)")
  x
}

for (i <- 1 to 10) {
  matFutures += ConcurrentContext.executeAsync(slowFoo(true))

  if ((j + 1) % maxConcurrentFutures == 0) {
    for (fut <- matFutures) {
      println("Awaiting...")
      Await.result(fut, timeout)
    }
    matFutures.clear
  }
  j += 1
}

// left over futures
for (fut <- matFutures) {
  Await.result(fut, timeout)
}
// exit
sys.exit

}
