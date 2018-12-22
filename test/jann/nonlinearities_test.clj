(ns jann.nonlinearities-test
  (:require
    [clojure.test :refer :all]
    [jann.nonlinearities :as nonlin]))
(use 'clojure.core.matrix)

(deftest test-sigmoid
  (testing "if sigmoid works"
    (is (= [0.5 1.0 0.0] (nonlin/sigmoid [0 10000 -10000])))))

(deftest test-sigmoid-derivative
  (testing "if sigmoid derivative works"
    (is (= [0.25 0.0 0.0] (nonlin/sigmoid-derivative [0 10000 -10000])))))