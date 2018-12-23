(ns jann.dataset
  (:require [clojure.java.io :as io]
            [clojure.edn :as edn])
  (:use [clojure.core.matrix])
  (:import (java.util Random ArrayList Collections)
           (clojure.lang RT)))

(defn load-edn
  "Load edn from an io/reader source (filename or io/resource)."
  [source]
  (try
    (with-open [r (io/reader (io/resource source))]
      (edn/read (java.io.PushbackReader. r)))
    (catch java.io.IOException e
      (printf "Couldn't open '%s': %s\n" source (.getMessage e)))
    (catch RuntimeException e
      (printf "Error parsing edn file '%s': %s\n" source (.getMessage e)))))

(defn load-dataset
  ([images-file labels-file]
   (let [images (load-edn images-file)
         labels (load-edn labels-file)]
     {:train {:images (subvec images (int (round (* 0.2 (count images)))))
              :labels (subvec labels (int (round (* 0.2 (count images)))))}
      :test  {:images (subvec images 0 (int (round (* 0.2 (count images)))))
              :labels (subvec labels 0 (int (round (* 0.2 (count images)))))}}))
  ([] (load-dataset "t10k-images.edn" "t10k-labels.edn")))

(defn deterministic-shuffle
  [^java.util.Collection coll seed]
  (let [al (ArrayList. coll)
        rng (Random. seed)]
    (Collections/shuffle al rng)
    (RT/vector (.toArray al))))

(defn shuffle-dataset [{:keys [train test]}]
  (let [nonce (rand-int 100000)]
    {:train {:images (deterministic-shuffle (:images train) nonce)
             :labels (deterministic-shuffle (:labels train) nonce)}
     :test  {:images (deterministic-shuffle (:images test) nonce)
             :labels (deterministic-shuffle (:labels test) nonce)}}))