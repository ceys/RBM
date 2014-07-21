package com.jd.ajmd.util

import scala.util.Random
import scala.math.exp

/**
 * Created by zhengchen on 14-7-21.
 */
object MathUtil {

  val rand = new Random(System.currentTimeMillis())

  def binomial(p: Double): Int = {
    if (p > rand.nextDouble())
      1
    else
      0
  }


  def multinomial(pa: Array[Double]): Array[Int] = {
    var sum = 0.0
    val cu = pa.map { a => sum += a; sum}
    p = rand.nextDouble()
    cu.map { a => if (a > p)}

  }


  def softMax(a: Array[Double]): Array[Double] = {
    val total = a.sum
    a.map(exp(_)/total)
  }


}
