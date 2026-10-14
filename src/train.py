import os
import numpy as np
import joblib
import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score)

from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import ( #type: ignore
    Input,
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed
)

from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

from preprocess import (
    load_data,
    create_cyclic_features,
    encode_and_split_features_targets
)

from utils import (
    evaluate_forecasting_model,
    log_forecasting_metrics,
    evaluate_anomaly_model,
    log_anomaly_metrics
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/energy_processed.csv"

SEQUENCE_LENGTH = 24

RANDOM_STATE = 42

os.makedirs("model", exist_ok=True)

# MLflow experiment
mlflow.set_experiment("Energy Consumption intelligence")


# ============================================================
# FORECASTING SEQUENCE CREATION
# ============================================================

def create_forecasting_sequences(
    X,
    y,
    sequence_length=24
):

    X_sequences = []
    y_sequences = []

    for i in range(
        sequence_length,
        len(X)
    ):

        X_sequences.append(
            X[i - sequence_length:i]
        )

        y_sequences.append(
            y[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# ============================================================
# ANOMALY SEQUENCE CREATION
# ============================================================

def create_anomaly_sequences(
    X,
    y,
    sequence_length=24
):

    X_sequences = []
    y_sequences = []

    for i in range(
        sequence_length,
        len(X)
    ):

        # 24 previous observations
        X_sequences.append(
            X[i - sequence_length:i]
        )

        # Get anomaly labels for the same
        # 24 observations
        anomaly_window = y[
            i - sequence_length:i
        ]

        # Sequence is anomalous if
        # ANY observation is anomalous
        sequence_label = int(
            np.max(anomaly_window) > 0
        )

        y_sequences.append(
            sequence_label
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# ============================================================
# BUILD LSTM FORECASTING MODEL
# ============================================================

def build_forecasting_model(
    n_features,
    lstm_units,
    dense_units,
    dropout
):

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                n_features
            )
        ),

        LSTM(
            lstm_units
        ),

        Dropout(
            dropout
        ),

        Dense(
            dense_units,
            activation="relu"
        ),

        Dense(
            1
        )

    ])

    return model


# ============================================================
# BUILD LSTM AUTOENCODER
# ============================================================

def build_autoencoder(
    n_features,
    encoder_units,
    decoder_units,
    dropout
):

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                n_features
            )
        ),

        # Encoder
        LSTM(
            encoder_units
        ),

        Dropout(
            dropout
        ),

        # Convert vector back to sequence
        RepeatVector(
            SEQUENCE_LENGTH
        ),

        # Decoder
        LSTM(
            decoder_units,
            return_sequences=True
        ),

        Dropout(
            dropout
        ),

        TimeDistributed(
            Dense(n_features)
        )

    ])

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n==============================================")
    print("ENERGY CONSUMPTION DEEP LEARNING PROJECT")
    print("==============================================")

    print("\nLoading dataset...")

    df = load_data(DATA_PATH)

    print(
        "Dataset shape:",
        df.shape
    )


    # ========================================================
    # 2. CREATE CYCLIC FEATURES
    # ========================================================

    print("\nCreating cyclic features...")

    df = create_cyclic_features(df)


    # ========================================================
    # 3. ENCODE + SPLIT FEATURES AND TARGETS
    # ========================================================

    print("\nEncoding categorical variables...")

    (
        X,
        y_energy,
        y_anomaly,
        processed_df
    ) = encode_and_split_features_targets(df)


    print(
        "Feature shape:",
        X.shape
    )
    feature_columns = X.columns.tolist()
    joblib.dump(
        feature_columns,
        "model/feature_columns.pkl"
    )
    print("Feature columns saved to model/feature_columns.pkl")

    print(
        "Energy target shape:",
        y_energy.shape
    )

    print(
        "Anomaly target shape:",
        y_anomaly.shape
    )


    # ========================================================
    # 4. CHRONOLOGICAL SPLIT
    # ========================================================

    print("\nCreating chronological data split...")

    n = len(X)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )


    # -------------------------
    # TRAIN
    # -------------------------

    X_train = X.iloc[
        :train_end
    ].values

    y_energy_train = y_energy.iloc[
        :train_end
    ].values

    y_anomaly_train = y_anomaly.iloc[
        :train_end
    ].values


    # -------------------------
    # VALIDATION
    # -------------------------

    X_validation = X.iloc[
        train_end:validation_end
    ].values

    y_energy_validation = y_energy.iloc[
        train_end:validation_end
    ].values

    y_anomaly_validation = y_anomaly.iloc[
        train_end:validation_end
    ].values


    # -------------------------
    # TEST
    # -------------------------

    X_test = X.iloc[
        validation_end:
    ].values

    y_energy_test = y_energy.iloc[
        validation_end:
    ].values

    y_anomaly_test = y_anomaly.iloc[
        validation_end:
    ].values


    print("\nData split completed")

    print(
        "Training:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_validation)
    )

    print(
        "Testing:",
        len(X_test)
    )


    # ========================================================
    # 5. SCALE FEATURES
    # ========================================================

    print("\nScaling features...")

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit ONLY on training data

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    print(
        "Scaler saved to model/scaler.pkl"
    )


    # ========================================================
    # 6. CREATE FORECASTING SEQUENCES
    # ========================================================

    print(
        "\nCreating forecasting sequences..."
    )

    (
        X_train_forecast,
        y_train_forecast
    ) = create_forecasting_sequences(
        X_train_scaled,
        y_energy_train,
        SEQUENCE_LENGTH
    )


    (
        X_validation_forecast,
        y_validation_forecast
    ) = create_forecasting_sequences(
        X_validation_scaled,
        y_energy_validation,
        SEQUENCE_LENGTH
    )


    (
        X_test_forecast,
        y_test_forecast
    ) = create_forecasting_sequences(
        X_test_scaled,
        y_energy_test,
        SEQUENCE_LENGTH
    )


    print(
        "Training sequence:",
        X_train_forecast.shape
    )

    print(
        "Validation sequence:",
        X_validation_forecast.shape
    )

    print(
        "Testing sequence:",
        X_test_forecast.shape
    )


    n_features = X_train_forecast.shape[2]


    # ========================================================
    # 7. FIVE LSTM FORECASTING EXPERIMENTS
    # ========================================================

    forecasting_configs = [

        {
            "run_name": "LSTM_Forecasting_Run_1",
            "lstm_units": 32,
            "dense_units": 16,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32
        },

        {
            "run_name": "LSTM_Forecasting_Run_2",
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32
        },

        {
            "run_name": "LSTM_Forecasting_Run_3",
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.3,
            "learning_rate": 0.0005,
            "epochs": 40,
            "batch_size": 32
        },

        {
            "run_name": "LSTM_Forecasting_Run_4",
            "lstm_units": 128,
            "dense_units": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32
        },

        {
            "run_name": "LSTM_Forecasting_Run_5",
            "lstm_units": 128,
            "dense_units": 64,
            "dropout": 0.3,
            "learning_rate": 0.0005,
            "epochs": 40,
            "batch_size": 64
        }

    ]


    print(
        "\n=============================================="
    )

    print(
        "STARTING 5 LSTM FORECASTING RUNS"
    )

    print(
        "=============================================="
    )

     #forecasting loop
    for config in forecasting_configs:

        print(
            "\n----------------------------------------------"
        )

        print(
            config["run_name"]
        )

        print(
            "----------------------------------------------"
        )


        # Build model
        forecasting_model = (
            build_forecasting_model(
                n_features=n_features,
                lstm_units=config["lstm_units"],
                dense_units=config["dense_units"],
                dropout=config["dropout"]
            )
        )


        forecasting_model.compile(

            optimizer=Adam(
                learning_rate=config["learning_rate"]
            ),

            loss="mse",

            metrics=["mae"]
        )


        # Early stopping
        early_stopping = EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True
        )


        # Start MLflow run
        with mlflow.start_run(
            run_name=config["run_name"]
        ):

            # Log parameters
            mlflow.log_params({

                "task": "energy_forecasting",

                "model": "LSTM",

                "sequence_length":
                    SEQUENCE_LENGTH,

                "lstm_units":
                    config["lstm_units"],

                "dense_units":
                    config["dense_units"],

                "dropout":
                    config["dropout"],

                "learning_rate":
                    config["learning_rate"],

                "epochs":
                    config["epochs"],

                "batch_size":
                    config["batch_size"]

            })


            # Train
            forecasting_model.fit(

                X_train_forecast,

                y_train_forecast,

                validation_data=(

                    X_validation_forecast,

                    y_validation_forecast

                ),

                epochs=config["epochs"],

                batch_size=config["batch_size"],

                callbacks=[
                    early_stopping
                ],

                verbose=1
            )


            # Predict
            y_pred = (
                forecasting_model.predict(
                    X_test_forecast,
                    verbose=0
                ).flatten()
            )


            # Evaluate
            metrics = (
                evaluate_forecasting_model(
                    y_test_forecast,
                    y_pred
                )
            )


            print(
                "\nForecasting Results"
            )

            print(
                "MAE:",
                metrics["MAE"]
            )

            print(
                "RMSE:",
                metrics["RMSE"]
            )

            print(
                "R2:",
                metrics["R2"]
            )


            # Log metrics
            log_forecasting_metrics(

                model_name="LSTM",

                params=config,

                mae=metrics["MAE"],

                rmse=metrics["RMSE"],

                r2=metrics["R2"]

            )


            # Log model
            mlflow.keras.log_model(

                forecasting_model,

                "forecasting_model"

            )


    # ========================================================
    # 8. CREATE ANOMALY SEQUENCES
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "CREATING ANOMALY DETECTION SEQUENCES"
    )

    print(
        "=============================================="
    )


    (
        X_train_anomaly,
        y_train_anomaly
    ) = create_anomaly_sequences(

        X_train_scaled,

        y_anomaly_train,

        SEQUENCE_LENGTH

    )


    (
        X_validation_anomaly,
        y_validation_anomaly
    ) = create_anomaly_sequences(

        X_validation_scaled,

        y_anomaly_validation,

        SEQUENCE_LENGTH

    )


    (
        X_test_anomaly,
        y_test_anomaly
    ) = create_anomaly_sequences(

        X_test_scaled,

        y_anomaly_test,

        SEQUENCE_LENGTH

    )


    print(
        "Anomaly training sequence:",
        X_train_anomaly.shape
    )

    print(
        "Anomaly validation sequence:",
        X_validation_anomaly.shape
    )

    print(
        "Anomaly testing sequence:",
        X_test_anomaly.shape
    )


    # ========================================================
    # 9. FIND NORMAL TRAINING SEQUENCES
    # ========================================================

    print(
        "\nPreparing normal sequences..."
    )


    # NOW y_train_anomaly IS 1-D:
    #
    # [0, 0, 1, 0, 0, ...]
    #
    # because create_anomaly_sequences()
    # already converted each 24-hour window
    # into ONE label.
    #
    # Therefore we DO NOT use:
    #
    # np.max(y_train_anomaly, axis=1)
    #
    # anymore.

    normal_train_mask = (
        y_train_anomaly == 0
    )


    X_normal_train = (
        X_train_anomaly[
            normal_train_mask
        ]
    )


    normal_validation_mask = (
        y_validation_anomaly == 0
    )


    X_normal_validation = (
        X_validation_anomaly[
            normal_validation_mask
        ]
    )


    print(
        "Normal training sequences:",
        X_normal_train.shape
    )

    print(
        "Normal validation sequences:",
        X_normal_validation.shape
    )


    if len(X_normal_train) == 0:

        raise ValueError(
            "No normal training sequences found."
        )


    if len(X_normal_validation) == 0:

        raise ValueError(
            "No normal validation sequences found."
        )


    # ========================================================
    # 10. FIVE AUTOENCODER EXPERIMENTS
    # ========================================================

    autoencoder_configs = [

        {
            "run_name":
                "LSTM_Autoencoder_Run_1",

            "encoder_units":
                32,

            "decoder_units":
                32,

            "dropout":
                0.2,

            "learning_rate":
                0.001,

            "epochs":
                30,

            "batch_size":
                32
        },

        {
            "run_name":
                "LSTM_Autoencoder_Run_2",

            "encoder_units":
                64,

            "decoder_units":
                64,

            "dropout":
                0.2,

            "learning_rate":
                0.001,

            "epochs":
                30,

            "batch_size":
                32
        },

        {
            "run_name":
                "LSTM_Autoencoder_Run_3",

            "encoder_units":
                64,

            "decoder_units":
                64,

            "dropout":
                0.3,

            "learning_rate":
                0.0005,

            "epochs":
                40,

            "batch_size":
                32
        },

        {
            "run_name":
                "LSTM_Autoencoder_Run_4",

            "encoder_units":
                128,

            "decoder_units":
                64,

            "dropout":
                0.2,

            "learning_rate":
                0.001,

            "epochs":
                30,

            "batch_size":
                32
        },

        {
            "run_name":
                "LSTM_Autoencoder_Run_5",

            "encoder_units":
                128,

            "decoder_units":
                128,

            "dropout":
                0.3,

            "learning_rate":
                0.0005,

            "epochs":
                40,

            "batch_size":
                64
        }

    ]


    print(
        "\n=============================================="
    )

    print(
        "STARTING 5 LSTM AUTOENCODER RUNS"
    )

    print(
        "=============================================="
    )


    # ========================================================
    # 11. AUTOENCODER TRAINING LOOP
    # ========================================================

    for config in autoencoder_configs:

        print(
            "\n----------------------------------------------"
        )

        print(
            config["run_name"]
        )

        print(
            "----------------------------------------------"
        )


        # Build model
        anomaly_model = (
            build_autoencoder(

                n_features=n_features,

                encoder_units=
                    config["encoder_units"],

                decoder_units=
                    config["decoder_units"],

                dropout=
                    config["dropout"]

            )
        )


        anomaly_model.compile(

            optimizer=Adam(
                learning_rate=config["learning_rate"]
            ),
            loss = "mse"


               

        )


        early_stopping = EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True

        )


        # MLflow
        with mlflow.start_run(

            run_name=config["run_name"]

        ):


            # Log parameters
            mlflow.log_params({

                "task":
                    "anomaly_detection",

                "model":
                    "LSTM_Autoencoder",

                "sequence_length":
                    SEQUENCE_LENGTH,

                "encoder_units":
                    config["encoder_units"],

                "decoder_units":
                    config["decoder_units"],

                "dropout":
                    config["dropout"],

                "learning_rate":
                    config["learning_rate"],

                "epochs":
                    config["epochs"],

                "batch_size":
                    config["batch_size"]

            })


            # ------------------------------------------------
            # TRAIN ONLY ON NORMAL DATA
            # ------------------------------------------------

            anomaly_model.fit(

                X_normal_train,

                X_normal_train,

                validation_data=(

                    X_normal_validation,

                    X_normal_validation

                ),

                epochs=config["epochs"],

                batch_size=config["batch_size"],

                callbacks=[
                    early_stopping
                ],

                verbose=1

            )


            # ------------------------------------------------
            # VALIDATION RECONSTRUCTION
            # ------------------------------------------------

            validation_reconstructed = (

                anomaly_model.predict(

                    X_validation_anomaly,

                    verbose=0

                )

            )


            # Reconstruction error
            validation_scores = np.mean(

                np.square(

                    X_validation_anomaly
                    -
                    validation_reconstructed

                ),

                axis=(1, 2)

            )
            print("Validation reconstruction scores calculated.")


            # ------------------------------------------------
            # THRESHOLD
            # ------------------------------------------------

            threshold_percentiles = [
                70,
                75,
                80,
                85,
                90,
                92,
                94,
                95,
                96,
                97,
                98,
                99
            ]
            best_threshold = None
            best_recall = -1
            best_precision = 0
            best_f1 = 0
            best_percentile = None

            min_precision = 0.35

            print("\nThreshold Tuning Results:")

            for percentile in  threshold_percentiles:
                candidate_threshold = np.percentile(
                    validation_scores,
                    percentile
                )
                candidate_predictions = (
                    validation_scores > candidate_threshold
                ).astype(int)
                candidate_precision = precision_score(
                    y_validation_anomaly,
                    candidate_predictions,
                    zero_division=0
                )
                candidate_recall = recall_score(
                    y_validation_anomaly,
                    candidate_predictions,
                    zero_division=0
                )
                candidate_f1 = f1_score(
                    y_validation_anomaly,
                    candidate_predictions,
                    zero_division=0
                )
                print(
                    f"Percentile={percentile}% | "

                    f"Threshold={candidate_threshold:.6f} | "

                    f"Precision={candidate_precision:.4f} | "

                    f"Recall={candidate_recall:.4f} | "

                    f"F1={candidate_f1:.4f}"


                )
                if candidate_precision >= min_precision:

                     if candidate_recall > best_recall:

                        best_recall = candidate_recall

                        best_precision = candidate_precision

                        best_f1 = candidate_f1

                        best_threshold = candidate_threshold

                        best_percentile = percentile

            if best_threshold is None:
                print("\nNo threshold statisfied minimum precision")
                print("\nselecting threshold with highest validation recall")
                best_recall = -1
                for percentile in threshold_percentiles:
                        candidate_threshold = np.percentile(validation_scores,percentile)
                        candidate_predictions = (
                            validation_scores > candidate_threshold
                        ).astype(int)   
                        candidate_precision = precision_score(y_validation_anomaly,candidate_predictions,zero_division=0)
                        candidate_recall = recall_score(y_validation_anomaly,candidate_predictions,zero_division=0)
                        candidate_f1 = f1_score(y_validation_anomaly,candidate_predictions,zero_division=0)

                        if candidate_recall > best_recall:

                            best_recall = candidate_recall

                            best_precision = candidate_precision

                            best_f1 = candidate_f1

                            best_threshold = candidate_threshold

                            best_percentile = percentile

            #final selected threshold
            threshold = float(best_threshold)
            joblib.dump(
                    threshold,
                    "model/anomaly_threshold.pkl"
                )
            
            print(
                    "anomaly_threshold saved to model/anomaly_threshold.pkl"
                )


            print("\n==============================================")

            print("BEST ANOMALY THRESHOLD SELECTED")

            print("==============================================")

            print("Threshold percentile:",best_percentile)
                    

            print("Threshold:",threshold)
    
            print("Validation Precision:", best_precision)

            print("Validation Recall:",best_recall)
   
            print("Validation F1:",best_f1)
    
    
            # ------------------------------------------------
            # TEST RECONSTRUCTION
            # ------------------------------------------------

            test_reconstructed = (

                anomaly_model.predict(

                    X_test_anomaly,

                    verbose=0

                )

            )


            # test Reconstruction error
            anomaly_scores = np.mean(

                np.square(

                    X_test_anomaly
                    -
                    test_reconstructed

                ),

                axis=(1, 2)

            )


            # ------------------------------------------------
            # ANOMALY CLASSIFICATION
            # ------------------------------------------------

            y_anomaly_pred = (

                anomaly_scores > threshold

            ).astype(int)


            # ------------------------------------------------
            # EVALUATION
            # ------------------------------------------------

            anomaly_metrics = (

                evaluate_anomaly_model(

                    y_test_anomaly,

                    y_anomaly_pred,

                    anomaly_scores

                )

            )


            print(
                "\nAnomaly Detection Results"
            )

            print(
                "Precision:",
                anomaly_metrics["precision"]
            )

            print(
                "Recall:",
                anomaly_metrics["recall"]
            )

            print(
                "F1 Score:",
                anomaly_metrics["f1_score"]
            )

            print(
                "ROC-AUC:",
                anomaly_metrics["roc_auc"]
            )


            # ------------------------------------------------
            # LOG METRICS
            # ------------------------------------------------

            log_anomaly_metrics(

                model_name=
                    "LSTM_Autoencoder",

                params={
                    **config,
                    "threshold_percentile": 
                          best_percentile,
                    "minimum_precision":
                        min_precision
                },

                precision=
                    anomaly_metrics["precision"],

                recall=
                    anomaly_metrics["recall"],

                f1=
                    anomaly_metrics["f1_score"],

                roc_auc=
                    anomaly_metrics["roc_auc"]

            )


            # Log threshold
            mlflow.log_metric(

                "anomaly_threshold",

                 float(threshold)

            )


           

            #to log validation results
            mlflow.log_metric(
               "validation_precision",
                float(best_precision)
            )

            mlflow.log_metric(
                "validation_recall",
                 float(best_recall)
            )

            mlflow.log_metric(
                "validation_f1",
                 float(best_f1)
            )

            mlflow.log_param(
                "selected_threshold_percentile",
                 best_percentile
            )

             # Log model
            mlflow.keras.log_model(
            
                anomaly_model,
            
                "anomaly_model"
            
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "ALL TRAINING RUNS COMPLETED SUCCESSFULLY!"
    )

    print(
        "=============================================="
    )

    print(
        "\nMLflow experiment:"
    )

    print(
        "Energy Consumption"
    )

    print(
        "\nTotal runs:"
    )

    print(
        "5 LSTM Forecasting + "
        "5 LSTM Autoencoder = 10 runs"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()