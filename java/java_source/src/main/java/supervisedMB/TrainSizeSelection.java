package supervisedMB;

import gnu.trove.map.TObjectFloatMap;
import gnu.trove.map.hash.TObjectFloatHashMap;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;

import java.io.*;
import java.util.*;

public class TrainSizeSelection {
    static void applyBLAST(double portion, double totalMatches, List<Comparison> comparisons, PrintWriter out, String info, Double initTime) {
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

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        long endTime = System.nanoTime();

        System.out.println("BLAST matches\t:\t" + finalMatches);
        System.out.println("BLAST comparisons\t:\t" + finalComparisons);
        System.out.println("BLAST PC\t:\t" + pc);
        System.out.println("BLAST PQ\t:\t" + pq);
        System.out.println("BLAST F-Measure\t:\t" + f1);

        double elapsedTime = ((double) (endTime - startTime) / 1_000_000_000) + initTime;

        out.println(info + "blast;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
    }

    static void applyReciprocalCNP(double totalMatches, float k, List<Comparison> comparisons, PrintWriter out, String info, Double initTime) {
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
        for (Map.Entry<String, Queue<Comparison>> entry : pairsPerEntity1.entrySet()) {
            Set<String> entitiesD2 = new HashSet<>();
            for (Comparison c : entry.getValue()) {
                entitiesD2.add(c.getEntity2());
            }
            topPairsPerEntity1.put(entry.getKey(), entitiesD2);
        }

        final Map<String, Set<String>> topPairsPerEntity2 = new HashMap<>();
        for (Map.Entry<String, Queue<Comparison>> entry : pairsPerEntity2.entrySet()) {
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

        double pc = finalMatches / totalMatches;
        double pq = finalMatches / (double) finalComparisons;
        double f1 = 2 * pc * pq / (pc + pq);
        long endTime = System.nanoTime();
        System.out.println("Reciprocal CNP matches\t:\t" + finalMatches);
        System.out.println("Reciprocal CNP comparisons\t:\t" + finalComparisons);
        System.out.println("Reciprocal CNP PC\t:\t" + pc);
        System.out.println("Reciprocal CNP PQ\t:\t" + pq);
        System.out.println("Reciprocal CNP F-Measure\t:\t" + f1);
        double elapsedTime = ((double) (endTime - startTime) / 1_000_000_000) + initTime;
        out.println(info + "RCNP;" + finalMatches + ";" + finalComparisons + ";" + pc + ";" + pq + ";" + f1 + ";" + elapsedTime);
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

    static Data loadData(String dname, int train_size, int conf_id, int run) throws IOException {
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

        Data data = new Data();
        data.comparisons = comparisons;
        data.retainedMatches = retainedMatches;
        data.num = num;
        data.initialMatches = initialMatches;
        data.runtime = elapsedTime;
        return data;
    }

    public static void main(String[] args) throws IOException {

        PrintWriter out = new PrintWriter(new File("/home/app/results/train_size_selection_java.csv"));
        out.println("dataset;train_size;conf_id;run;algorithm;matches;comparisons;recall;precision;F1;RT");

        HashMap<String, Stats> stats = load_stats("/home/app/results/01b_blocking_stats.csv");

        Properties props = new Properties();
        FileInputStream in = new FileInputStream("/home/app/config/config.ini");
        props.load(in);
        int num_runs = Integer.parseInt(props.getProperty("repetitions"));

        int[] train_set_sizes = {20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
        int blast_feat_id = 78;
        int rcnp_feat_id = 187;
        int original_feat_id = 128;

        String[] datasets = {"AbtBuy", "DblpAcm", "ScholarDblp", "AmazonGP", "ImdbTmdb", "ImdbTvdb", "TmdbTvdb", "Movies", "WalmartAmazon"};
        ArrayList<String> dnames = new ArrayList<>(Arrays.asList(datasets));

        for (String dname : stats.keySet()) {
            if (dnames.contains(dname)) {
                for (int train_size : train_set_sizes) {
                    for (int run = 0; run < num_runs; run++) {
                        System.out.println("\n\nCurrent dataset\t:\t" + dname);
                        int noOfEntities = stats.get(dname).numProfiles;
                        double blockSizes = stats.get(dname).blockSizes;

                        Data data = loadData(dname, train_size, original_feat_id, run);

                        double pc = data.retainedMatches / (double) data.initialMatches;
                        double pq = data.retainedMatches / (double) data.comparisons.size();
                        double f1 = 2 * pc * pq / (pc + pq);
                        String info = dname + ";" + train_size + ";" + original_feat_id + ";" + run + ";";
                        out.println(info + "bcl;" + data.retainedMatches + ";" + data.comparisons.size() + ";" + pc + ";" + pq + ";" + f1 + ";" + data.runtime);


                        data = loadData(dname, train_size, blast_feat_id, run);
                        info = dname + ";" + train_size + ";" + blast_feat_id + ";" + run + ";";
                        applyBLAST(0.35, data.initialMatches, data.comparisons, out, info, data.runtime);


                        data = loadData(dname, train_size, rcnp_feat_id, run);
                        info = dname + ";" + train_size + ";" + rcnp_feat_id + ";" + run + ";";
                        applyReciprocalCNP(data.initialMatches, (float) Math.max(1, blockSizes / ((float) noOfEntities)), data.comparisons, out, info, data.runtime);
                    }
                }
            }
        }
        out.close();
    }
}
