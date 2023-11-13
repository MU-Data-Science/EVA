#!/usr/bin/env bash
EMAIL="spn8y@umsystem.edu"

echo "Completed processing. Sending out an email notification."
mail -s "Completed process notification!" ${EMAIL}
