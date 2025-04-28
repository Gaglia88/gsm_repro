package supervisedMB;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.json.*;

public class Dataset {
    public String name;

    public Dataset(String name) {
        this.name = name;
    }

    static ArrayList<String> getNames(ArrayList<Dataset> datasets) {
        ArrayList<String> names = new ArrayList<String>();
        for (Dataset dataset : datasets) {
            names.add(dataset.name);
        }
        return names;
    }

    static ArrayList<Dataset> loadDatasets(String datasetsFilePath, String dtype) {
        String contents = null;
        ArrayList<Dataset> datasets = new ArrayList<>();
        try {
            contents = new String(Files.readAllBytes(Paths.get(datasetsFilePath)), StandardCharsets.UTF_8);
        } catch (Exception e) {
            System.out.print("Cannot read the dataset file.");
        }
        if (contents != null) {
            try {
                JSONArray datasetsJSON = new JSONArray(contents);
                for (int i = 0; i < datasetsJSON.length(); i++) {
                    JSONObject datasetJSON = datasetsJSON.getJSONObject(i);
                    if (dtype.equals("all") || datasetJSON.getString("type").equals(dtype)) {
                        datasets.add(new Dataset(datasetJSON.getString("name")));
                    }
                }
            } catch (Exception e) {
                System.out.print("Cannot read the dataset file.");
            }
        }
        return datasets;
    }
}
