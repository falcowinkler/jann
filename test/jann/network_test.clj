(ns jann.network-test
  (:require
    [clojure.test :refer :all]
    [jann.network :as n])
  (:use [clojure.core.matrix]))

(deftest make-network-test
  (testing "if network initialization works"
    (is (= {:weights [(array [[0.0 0.0]
                              [0.0 0.0]
                              [0.0 0.0]])
                      (array [[0.0 0.0 0.0]])]
            :biases  [(array [0.0 0.0 0.0]) (array [0.0])]}
           (n/make-network [2 3 1] identity)))))

(deftest inference-test
  (testing "if inference works"
    (is (= (array [0.5])
           (n/infer [0 0] (n/make-network [2 3 1] identity))))
    (is (= (array [1.0])
           (n/infer [0 0]
                    {:weights [(array [[0 0]
                                       [0 0]
                                       [0 0]])
                               (array [[0 0 0]])]
                     :biases  [(array [0 0 0]) (array [10000])]})))))

(deftest argmax-test
  (testing "if argmax works"
    (is (= 2 (n/argmax (array [1 2 2341 12 93  12 1 1 2 5 3 2]))))))