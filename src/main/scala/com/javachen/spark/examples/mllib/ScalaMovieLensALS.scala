package com.javachen.spark.examples.mllib

import java.util.Random

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * see:https://github.com/mohit-shrma/RandomSamples/blob/d9f1117bc21bb09d9fa858bc6d95e08e753e6fa0/SparkScala/CollabFilter/src/main/scala/MovieLensALS.scala
 */
object ScalaMovieLensALS {

  def main(args: Array[String]) {

    //import org.apache.log4j.{Logger,Level}
    //Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    //Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 2) {
      println("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g --class com.javachen.spark.examples.mllib.ScalaMovieLensALS " +
        "target/scala-*/movielens-als-ssembly-*.jar movieLensHomeDir personalRatingsFile")
      sys.exit(1)
    }

    // set up environment
    val conf = new SparkConf().setAppName("ScalaMovieLensALS")
    val sc = new SparkContext(conf)

    // load ratings and movie titles
    val ratings = sc.textFile(args(0) + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    val movies = sc.textFile(args(0) + "/movies.dat").map { line =>
      val fields = line.split("::")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect().toMap

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()

    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

    //get ratings of user on top 50 popular movies
    val mostRatedMovieIds = ratings.map(_._2.product) //extract movieId
      .countByValue //count ratings per movie
      .toSeq //convert map to seq
      .sortBy(-_._2) //sort by rating count in decreasing order
      .take(50) //take 50 most rated
      .map(_._1) //get movie ids

    val random = new Random(0)
    val selectedMovies = mostRatedMovieIds.filter(x => random.nextDouble() < 0.2)
      .map(x => (x, movies(x)))
      .toSeq
    val myRatings = elicitateRatings(selectedMovies)
    //convert received ratings to RDD[Rating], now this can be worked in parallel
    val myRatingsRDD = sc.parallelize(myRatings)

    // split ratings into train (60%), validation (20%), and test (20%) based on the
    // last digit of the timestamp, add myRatings to train, and cache them

    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6).values.union(myRatingsRDD).repartition(numPartitions).cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8).values.repartition(numPartitions).cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()

    println(s"Training: $numTraining, validation: $numValidation, test: $numTest")

    // train models and evaluate them on the validation set
    val ranks = List(8, 10, 12)
    val lambdas = List(0.1, 1.0, 10.0)
    val numIterations = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIterations) {
      //learn model for these parameter
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation)
      println(s"RMSE (validation) = $validationRmse for the model trained with rank = $rank , lambda = $lambda , and numIter = $numIter .")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // evaluate the best model on the test set
    val testRmse = computeRmse(bestModel.get, test)
    println(s"The best model was trained with rank = $bestRank and lambda = $bestLambda , and numIter = $bestNumIter , and its RMSE on the test set is $testRmse .")

    //find best movies for the user
    val myRatedMovieIds = myRatings.map(_.product).toSet
    //generate candidates after taking out already rated movies
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get.predict(candidates.map((0, _))).collect.sortBy(-_.rating).take(50)
    var i = 1
    println("Movies recommendation for you: ")
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }

    // create a naive baseline and compare it with the best model
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // clean up
    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]) = {
    val usersProducts = data.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }

    val ratesAndPreds = data.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions).sortByKey()

    math.sqrt(ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean())
  }

  /** Elicitate ratings from commandline **/
  def elicitateRatings(movies: Seq[(Int, String)]) = {
    val prompt = "Please rate following movie (1-5(best), or 0 if not seen):"
    println(prompt)
    val ratings = movies.flatMap { x =>

      var rating: Option[Rating] = None
      var valid = false

      while (!valid) {
        print(x._2 + ": ")
        try {
          val r = Console.readInt
          if (r < 0 || r > 5) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }

      rating match {
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }

    } //end flatMap

    if (ratings.isEmpty) {
      error("No rating provided")
    } else {
      ratings
    }

  }

}
