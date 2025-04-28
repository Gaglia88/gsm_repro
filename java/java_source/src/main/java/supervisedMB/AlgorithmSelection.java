/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package supervisedMB;

import gnu.trove.map.TObjectFloatMap;
import gnu.trove.map.hash.TObjectFloatHashMap;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;

/**
 * @author Georgios
 */
public class AlgorithmSelection {

    static void applyBLAST(double portion, double totalMatches, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        TObjectFloatMap<String> maxWeightPerEntity1 = new TObjectFloatHashMap<>();
        TObjectFloatMap<String> maxWeightPerEntity2 = new TObjectFloatHashMap<>();
        for (Comparison c : comparisons) {
            double maxWeight1 = maxWeightPerEntity1.get(c.getEntity1());
            if (maxWeight1 < c.getProb()) {
                maxWeightPerEntity1.put(c.getEntity1(), c.getProb());
            }

            double maxWeight2 = maxWeightPerEntity2.get(c.getEntity2());
            if (maxWeight2 < c.getProb()) {
                maxWeightPerEntity2.put(c.getEntity2(), c.getProb());
            }
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            double maxWeight1 = maxWeightPerEntity1.get(c.getEntity1());
            double maxWeight2 = maxWeightPerEntity2.get(c.getEntity2());
            double threshold = portion * (maxWeight1 + maxWeight2);// / 3;
            if (threshold <= c.getProb()) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("BLAST matches\t:\t" + finalMatches);
        System.out.println("BLAST comparisons\t:\t" + finalComparisons);
        System.out.println("BLAST PC\t:\t" + pc);
        System.out.println("BLAST PQ\t:\t" + pq);
        System.out.println("BLAST F-Measure\t:\t" + f1);

        out.println(info + "blast;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyCEP(double totalMatches, float K, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        final Queue<Comparison> topKEdges = new PriorityQueue<>((int) (2 * K), new IncComparisonWeightComparator());

        float minimumWeight = 0;
        for (Comparison c : comparisons) {
            if (minimumWeight <= c.getProb()) {
                topKEdges.add(c);
                if (K < topKEdges.size()) {
                    final Comparison lastComparison = topKEdges.poll();
                    minimumWeight = lastComparison.getProb();
                }
            }
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : topKEdges) {
            if (c.isMatch()) {
                finalMatches++;
            }
            finalComparisons++;
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("CEP matches\t:\t" + finalMatches);
        System.out.println("CEP comparisons\t:\t" + finalComparisons);
        System.out.println("CEP PC\t:\t" + pc);
        System.out.println("CEP PQ\t:\t" + pq);
        System.out.println("CEP F-Measure\t:\t" + f1);
        out.println(info + "CEP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyCNP(double totalMatches, float k, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        final Map<String, Queue<Comparison>> pairsPerEntity1 = new HashMap<>();
        final Map<String, Queue<Comparison>> pairsPerEntity2 = new HashMap<>();
        for (Comparison c : comparisons) {
            Queue<Comparison> queue1 = pairsPerEntity1.get(c.getEntity1());
            if (queue1 == null) {
                queue1 = new PriorityQueue<>((int) (2 * k), new IncComparisonWeightComparator());
                pairsPerEntity1.put(c.getEntity1(), queue1);
            }
            queue1.add(c);
            if (k < queue1.size()) {
                queue1.poll();
            }

            Queue<Comparison> queue2 = pairsPerEntity2.get(c.getEntity2());
            if (queue2 == null) {
                queue2 = new PriorityQueue<>((int) (2 * k), new IncComparisonWeightComparator());
                pairsPerEntity2.put(c.getEntity2(), queue2);
            }
            queue2.add(c);
            if (k < queue2.size()) {
                queue2.poll();
            }
        }

        final Map<String, Set<String>> topPairsPerEntity1 = new HashMap<>();
        for (Entry<String, Queue<Comparison>> entry : pairsPerEntity1.entrySet()) {
            Set<String> entitiesD2 = new HashSet<>();
            for (Comparison c : entry.getValue()) {
                entitiesD2.add(c.getEntity2());
            }
            topPairsPerEntity1.put(entry.getKey(), entitiesD2);
        }

        final Map<String, Set<String>> topPairsPerEntity2 = new HashMap<>();
        for (Entry<String, Queue<Comparison>> entry : pairsPerEntity2.entrySet()) {
            Set<String> entitiesD1 = new HashSet<>();
            for (Comparison c : entry.getValue()) {
                entitiesD1.add(c.getEntity1());
            }
            topPairsPerEntity2.put(entry.getKey(), entitiesD1);
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            Set<String> p1Candidates = topPairsPerEntity1.get(c.getEntity1());
            Set<String> p2Candidates = topPairsPerEntity2.get(c.getEntity2());
            if (p1Candidates.contains(c.getEntity2()) || p2Candidates.contains(c.getEntity1())) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("CNP matches\t:\t" + finalMatches);
        System.out.println("CNP comparisons\t:\t" + finalComparisons);
        System.out.println("CNP PC\t:\t" + pc);
        System.out.println("CNP PQ\t:\t" + pq);
        System.out.println("CNP F-Measure\t:\t" + f1);
        out.println(info + "CNP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyReciprocalCNP(double totalMatches, float k, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        final Map<String, Queue<Comparison>> pairsPerEntity1 = new HashMap<>();
        final Map<String, Queue<Comparison>> pairsPerEntity2 = new HashMap<>();
        for (Comparison c : comparisons) {
            Queue<Comparison> queue1 = pairsPerEntity1.get(c.getEntity1());
            if (queue1 == null) {
                queue1 = new PriorityQueue<>((int) (2 * k), new IncComparisonWeightComparator());
                pairsPerEntity1.put(c.getEntity1(), queue1);
            }
            queue1.add(c);
            if (k < queue1.size()) {
                queue1.poll();
            }

            Queue<Comparison> queue2 = pairsPerEntity2.get(c.getEntity2());
            if (queue2 == null) {
                queue2 = new PriorityQueue<>((int) (2 * k), new IncComparisonWeightComparator());
                pairsPerEntity2.put(c.getEntity2(), queue2);
            }
            queue2.add(c);
            if (k < queue2.size()) {
                queue2.poll();
            }
        }

        final Map<String, Set<String>> topPairsPerEntity1 = new HashMap<>();
        for (Entry<String, Queue<Comparison>> entry : pairsPerEntity1.entrySet()) {
            Set<String> entitiesD2 = new HashSet<>();
            for (Comparison c : entry.getValue()) {
                entitiesD2.add(c.getEntity2());
            }
            topPairsPerEntity1.put(entry.getKey(), entitiesD2);
        }

        final Map<String, Set<String>> topPairsPerEntity2 = new HashMap<>();
        for (Entry<String, Queue<Comparison>> entry : pairsPerEntity2.entrySet()) {
            Set<String> entitiesD1 = new HashSet<>();
            for (Comparison c : entry.getValue()) {
                entitiesD1.add(c.getEntity1());
            }
            topPairsPerEntity2.put(entry.getKey(), entitiesD1);
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            Set<String> p1Candidates = topPairsPerEntity1.get(c.getEntity1());
            Set<String> p2Candidates = topPairsPerEntity2.get(c.getEntity2());
            if (p1Candidates.contains(c.getEntity2()) && p2Candidates.contains(c.getEntity1())) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("Reciprocal CNP matches\t:\t" + finalMatches);
        System.out.println("Reciprocal CNP comparisons\t:\t" + finalComparisons);
        System.out.println("Reciprocal CNP PC\t:\t" + pc);
        System.out.println("Reciprocal CNP PQ\t:\t" + pq);
        System.out.println("Reciprocal CNP F-Measure\t:\t" + f1);
        out.println(info + "RCNP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyReciprocalWNP(double totalMatches, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        final Map<String, List<Float>> weightsPerEntity1 = new HashMap<>();
        final Map<String, List<Float>> weightsPerEntity2 = new HashMap<>();
        for (Comparison c : comparisons) {
            List<Float> list1 = weightsPerEntity1.get(c.getEntity1());
            if (list1 == null) {
                list1 = new ArrayList<>();
                weightsPerEntity1.put(c.getEntity1(), list1);
            }
            list1.add(c.getProb());

            List<Float> list2 = weightsPerEntity2.get(c.getEntity2());
            if (list2 == null) {
                list2 = new ArrayList<>();
                weightsPerEntity2.put(c.getEntity2(), list2);
            }
            list2.add(c.getProb());
        }

        TObjectFloatMap<String> avWeightPerEntity1 = new TObjectFloatHashMap<>();
        for (Entry<String, List<Float>> entry : weightsPerEntity1.entrySet()) {
            float avWeight = 0;
            for (Float w : entry.getValue()) {
                avWeight += w;
            }
            avWeight /= entry.getValue().size();
            avWeightPerEntity1.put(entry.getKey(), avWeight);
        }

        TObjectFloatMap<String> avWeightPerEntity2 = new TObjectFloatHashMap<>();
        for (Entry<String, List<Float>> entry : weightsPerEntity2.entrySet()) {
            float avWeight = 0;
            for (Float w : entry.getValue()) {
                avWeight += w;
            }
            avWeight /= entry.getValue().size();
            avWeightPerEntity2.put(entry.getKey(), avWeight);
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            float avWeight1 = avWeightPerEntity1.get(c.getEntity1());
            float avWeight2 = avWeightPerEntity2.get(c.getEntity2());
            if (avWeight1 <= c.getProb() && avWeight2 <= c.getProb()) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("Reciprocal WNP matches\t:\t" + finalMatches);
        System.out.println("Reciprocal WNP comparisons\t:\t" + finalComparisons);
        System.out.println("Reciprocal WNP PC\t:\t" + pc);
        System.out.println("Reciprocal WNP PQ\t:\t" + pq);
        System.out.println("Reciprocal WNP F-Measure\t:\t" + f1);
        out.println(info + "RWNP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyWEP(float totalMatches, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        float avWeight = 0;
        for (Comparison c : comparisons) {
            avWeight += c.getProb();
        }
        avWeight /= comparisons.size();

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            if (avWeight <= c.getProb()) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("New WEP matches\t:\t" + finalMatches);
        System.out.println("New WEP comparisons\t:\t" + finalComparisons);
        System.out.println("New WEP PC\t:\t" + pc);
        System.out.println("New WEP PQ\t:\t" + pq);
        System.out.println("New WEP F-Measure\t:\t" + f1);
        out.println(info + "WEP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyWNP(float totalMatches, List<Comparison> comparisons, PrintWriter out, String info, long initTime) {
        long startTime = System.nanoTime();
        final Map<String, List<Float>> weightsPerEntity1 = new HashMap<>();
        final Map<String, List<Float>> weightsPerEntity2 = new HashMap<>();
        for (Comparison c : comparisons) {
            List<Float> list1 = weightsPerEntity1.get(c.getEntity1());
            if (list1 == null) {
                list1 = new ArrayList<>();
                weightsPerEntity1.put(c.getEntity1(), list1);
            }
            list1.add(c.getProb());

            List<Float> list2 = weightsPerEntity2.get(c.getEntity2());
            if (list2 == null) {
                list2 = new ArrayList<>();
                weightsPerEntity2.put(c.getEntity2(), list2);
            }
            list2.add(c.getProb());
        }

        TObjectFloatMap<String> avWeightPerEntity1 = new TObjectFloatHashMap<>();
        for (Entry<String, List<Float>> entry : weightsPerEntity1.entrySet()) {
            float avWeight = 0;
            for (Float w : entry.getValue()) {
                avWeight += w;
            }
            avWeight /= entry.getValue().size();
            avWeightPerEntity1.put(entry.getKey(), avWeight);
        }

        TObjectFloatMap<String> avWeightPerEntity2 = new TObjectFloatHashMap<>();
        for (Entry<String, List<Float>> entry : weightsPerEntity2.entrySet()) {
            float avWeight = 0;
            for (Float w : entry.getValue()) {
                avWeight += w;
            }
            avWeight /= entry.getValue().size();
            avWeightPerEntity2.put(entry.getKey(), avWeight);
        }

        int finalMatches = 0;
        int finalComparisons = 0;
        for (Comparison c : comparisons) {
            float avWeight1 = avWeightPerEntity1.get(c.getEntity1());
            float avWeight2 = avWeightPerEntity2.get(c.getEntity2());
            if (avWeight1 <= c.getProb() || avWeight2 <= c.getProb()) {
                if (c.isMatch()) {
                    finalMatches++;
                }
                finalComparisons++;
            }
        }

        long endTime = System.nanoTime();
        double elapsedTime = (double) ((endTime - startTime) + initTime) / 1_000_000_000;

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        System.out.println("WNP matches\t:\t" + finalMatches);
        System.out.println("WNP comparisons\t:\t" + finalComparisons);
        System.out.println("WNP PC\t:\t" + pc);
        System.out.println("WNP PQ\t:\t" + pq);
        System.out.println("WNP F-Measure\t:\t" + f1);
        out.println(info + "WNP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static HashMap<String, Stats> load_stats(String path) throws IOException {
        HashMap<String, Stats> data = new HashMap<String, Stats>();

        final BufferedReader reader = new BufferedReader(new FileReader(path));
        String line = reader.readLine();//attributes line
        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) {
                String[] res = line.split(",");
                String dname = res[0];
                int numProfiles = Integer.parseInt(res[1]);
                double blockSizes = Double.parseDouble(res[2]);
                data.put(dname, new Stats(numProfiles, blockSizes));
            }
        }
        reader.close();

        return data;
    }

    public static void main(String[] args) throws IOException {
        Properties props = new Properties();
        FileInputStream in = new FileInputStream("/home/app/config/config.ini");
        props.load(in);
        int num_runs = Integer.parseInt(props.getProperty("repetitions"));
        ArrayList<String> dnames = Dataset.getNames(Dataset.loadDatasets("/home/app/datasets/datasets.json", "clean"));
        int train_size = Integer.parseInt(props.getProperty("train_set_size"));;
        int conf_id = Integer.parseInt(props.getProperty("supmb_original_set_id"));;

        PrintWriter out = new PrintWriter(new File("/home/app/results/algorithm_selection_java.csv"));
        out.println("dataset;train_size;conf_id;run;algorithm;matches;comparisons;recall;precision;F1;RT");

        HashMap<String, Stats> stats = load_stats("/home/app/results/01b_blocking_stats.csv");

        for (String dname : stats.keySet()) {
            for (int run = 0; run < num_runs; run++) {
                if (dnames.contains(dname)) {
                    System.out.println("\n\nCurrent dataset\t:\t" + dname);
                    int noOfEntities = stats.get(dname).numProfiles;
                    double blockSizes = stats.get(dname).blockSizes;

                    String path = "/home/app/probabilities/" + dname + "/" + train_size + "/" + dname + "_fs" + conf_id + "_run" + run + ".parquet";
                    Path file = new Path(path);
                    ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(file).build();
                    GenericRecord record;
                    List<Comparison> comparisons = new ArrayList<>();
                    int initialMatches = 0;
                    int retainedMatches = 0;
                    int num = 0;


                    long startTime = System.nanoTime();
                    while ((record = reader.read()) != null) {
                        float posProb = Float.parseFloat(record.get("p_match").toString());
                        boolean predictedMatch = Integer.parseInt(record.get("pred").toString()) == 1;
                        boolean isMatch = Integer.parseInt(record.get("is_match").toString()) == 1;
                        if (isMatch) {
                            initialMatches++;
                        }
                        if (predictedMatch) {
                            if (isMatch) {
                                retainedMatches++;
                            }
                        }
                        if (posProb >= 0.5) {
                            comparisons.add(new Comparison(isMatch, posProb, record.get("p1").toString(), record.get("p2").toString()));
                        }
                        num += 1;
                    }
                    reader.close();
                    long endTime = System.nanoTime();
                    double elapsedTime = (double) (endTime - startTime) / 1_000_000_000;

                    double pc = retainedMatches / (double) initialMatches;
                    double pq = retainedMatches / (double) comparisons.size();
                    double f1 = 2 * pc * pq / (pc + pq);

                    System.out.println("Initial comparisons\t:\t" + num);
                    System.out.println("Initial matches\t:\t" + initialMatches);
                    System.out.println("Retained comparisons\t:\t" + comparisons.size());
                    System.out.println("Retained matches\t:\t" + retainedMatches);
                    System.out.println("Original WEP PC\t:\t" + pc);
                    System.out.println("Original WEP PQ\t:\t" + pq);
                    System.out.println("Original WEP F-Measure\t:\t" + f1);

                    String info = dname + ";" + train_size + ";" + conf_id + ";" + run + ";";

                    out.println(info + "bcl;" + retainedMatches + ";" + comparisons.size() + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);

                    applyWEP(initialMatches, comparisons, out, info, endTime - startTime);
                    applyWNP(initialMatches, comparisons, out, info, endTime - startTime);
                    applyReciprocalWNP(initialMatches, comparisons, out, info, endTime - startTime);
                    applyBLAST(0.35, initialMatches, comparisons, out, info, endTime - startTime);
                    applyCEP(initialMatches, (float) blockSizes / 2.0f, comparisons, out, info, endTime - startTime);
                    applyCNP(initialMatches, (float) Math.max(1, blockSizes / ((float) noOfEntities)), comparisons, out, info, endTime - startTime);
                    applyReciprocalCNP(initialMatches, (float) Math.max(1, blockSizes / ((float) noOfEntities)), comparisons, out, info, endTime - startTime);
                }
            }
        }
        out.close();
    }
}
