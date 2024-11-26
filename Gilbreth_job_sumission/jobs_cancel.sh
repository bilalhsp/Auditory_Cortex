for job_id in {6686783..6686873}; do
    scancel $job_id
done

for job_id in {6686783..6686873}; do scancel $job_id; done
