OPENAI_API_KEY="${OPENAI_API_KEY}"

python analysis_framework_generator.py \
--type_of_anomaly "Contextual Anomaly \
--temperature 0.7 \
--paper_directory "papers" \
--output_file "analysis_framework_contextual.txt" \
--api_key $OPENAI_API_KEY

python analysis_framework_generator.py \
--type_of_anomaly "Structural Anomaly" \
--temperature 0.7 \
--paper_directory "papers" \
--output_file "analysis_framework_structural.txt" \
--api_key $OPENAI_API_KEY

python analysis_framework_generator.py \
--type_of_anomaly "Mixed Anomaly" \
--temperature 0.7 \
--paper_directory "papers" \
--output_file "analysis_framework_mixed.txt" \
--api_key $OPENAI_API_KEY