package com.sushant.example

import jep.Jep

object JepAddExample extends App {
  val jep = new Jep()
  println("Running from Scala")
  jep.runScript("src/main/python/add.py")
  var x = 0
  for (x <- 3 to 7) {
    jep.eval(s"c = test($x)")
  }
  // There are multiple ways to evaluate. Let us demonstrate them:
}
