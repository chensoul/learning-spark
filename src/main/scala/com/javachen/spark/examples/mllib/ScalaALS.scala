package com.javachen.spark.examples.mllib

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import scala.sys.process._

object ScalaALS {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Scala Collaborative Filtering Example"))

    // 加载并解析数据
    val data = sc.textFile("data/mllib/als/test.data")

    val ratings = data.map(_.split(',') match { case Array(user, product, rate) =>
      Rating(user.toInt, product.toInt, rate.toDouble)
    })

    //使用ALS训练数据建立推荐模型

    val rank = 12
    val lambda = 0.1
    val numIterations = 20
    val numPartitions=4

    val training = ratings.values.repartition(numPartitions).cache()
    val model = ALS.train(training, rank, numIterations, lambda)

    //从 ratings 中获得只包含用户和商品的数据集
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    //使用推荐模型对用户商品进行预测评分，得到预测评分的数据集
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    //将真实评分数据集与预测评分数据集进行合并
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions).sortByKey()

    //然后计算均方差
    val MSE =ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    println("Mean Squared Error = " + MSE)

    //确保只生成一个文件，并按用户排序
    val formatedRatesAndPreds = ratesAndPreds.repartition(1).sortBy(_._1).map({
      case ((user, product), (rate, pred)) => (user + "\t" + product + "\t" + rate + "\t" + pred)
    })

    "hadoop fs -rm -r /tmp/user_goods_rates".!
    formatedRatesAndPreds.saveAsTextFile("/tmp/user_goods_rates")

    //排序取10条，限制结果集为5
    predictions.map({ case ((user, product), rate) => (user, (product, rate)) }).groupByKey().map(t=>(t._1,t._2.toList.sortBy(x=> - x._2).take(10))).take(5)

  }

}
