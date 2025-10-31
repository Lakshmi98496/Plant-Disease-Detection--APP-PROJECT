import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

public class CsvResultsJavaFX extends Application {

    public static class ResultRow {
        private final String filename;
        private final String predictedClass;

        public ResultRow(String filename, String predictedClass) {
            this.filename = filename;
            this.predictedClass = predictedClass;
        }

        public String getFilename() { return filename; }
        public String getPredictedClass() { return predictedClass; }
    }

    @Override
    public void start(Stage primaryStage) {
        TableView<ResultRow> table = new TableView<>();
        TableColumn<ResultRow, String> filenameCol = new TableColumn<>("Filename");
        filenameCol.setCellValueFactory(new PropertyValueFactory<>("filename"));

        TableColumn<ResultRow, String> predictedCol = new TableColumn<>("Predicted Class");
        predictedCol.setCellValueFactory(new PropertyValueFactory<>("predictedClass"));

        table.getColumns().add(filenameCol);
        table.getColumns().add(predictedCol);

        ObservableList<ResultRow> data = FXCollections.observableArrayList();

        try (BufferedReader br = new BufferedReader(new FileReader("results.csv"))) {
            // Skip the header
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] fields = line.split(",");
                if (fields.length == 2) {
                    data.add(new ResultRow(fields[0].trim(), fields[1].trim()));
                }
            }
        } catch (IOException e) {
            System.out.println("CSV read error: " + e.getMessage());
        }

        table.setItems(data);

        VBox vbox = new VBox(table);
        Scene scene = new Scene(vbox, 600, 400);

        primaryStage.setScene(scene);
        primaryStage.setTitle("CSV Results Viewer");
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}