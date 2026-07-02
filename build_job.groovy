pipeline {
  agent any

  options {
    timestamps()
  }

  stages {
    stage('Trigger remote Jenkins job and stream console log') {
      steps {
        script {
          /*
           * Update these values for your environment.
           */
          String remoteJenkinsName = 'remoteJenkins'
          String remoteJobName = 'A'
          String remoteCredentialId = 'remote-jenkins-api-token'

          /*
           * Map parameters are recommended by the Parameterized Remote Trigger Plugin.
           * Remove this map or leave it empty if the remote job does not need parameters.
           */
          Map<String, Object> remoteParameters = [
            ENV: 'dev',
            FOO: 'bar'
          ]

          /*
           * Log polling interval in seconds.
           * Do not set this too low, otherwise it may create unnecessary load on the remote Jenkins.
           */
          int logPollSeconds = 2

          def remoteBuildHandle = null
          String remoteBuildUrl = null

          withCredentials([
            usernamePassword(
              credentialsId: remoteCredentialId,
              usernameVariable: 'REMOTE_JENKINS_USER',
              passwordVariable: 'REMOTE_JENKINS_TOKEN'
            )
          ]) {
            /*
             * Trigger the remote Jenkins job.
             *
             * blockBuildUntilComplete: false
             *   The step returns after the remote job is triggered.
             *
             * enhancedLogging: false
             *   The plugin will not copy the remote console log.
             *   This pipeline streams the log manually with curl.
             *
             * shouldNotFailBuild: true
             *   The trigger step itself will not fail this pipeline if the remote job fails.
             *   The final result is checked explicitly at the end.
             */
            remoteBuildHandle = triggerRemoteJob(
              remoteJenkinsName: remoteJenkinsName,
              job: remoteJobName,
              parameters: remoteParameters,
              auth: [
                $class: 'CredentialsAuth',
                credentials: remoteCredentialId
              ],

              blockBuildUntilComplete: false,
              enhancedLogging: false,
              shouldNotFailBuild: true,

              /*
               * If this pipeline is aborted, the triggered remote job will also be aborted.
               * Change this to false if you want the remote job to continue running.
               */
              abortTriggeredJob: true,

              /*
               * This is the plugin polling interval for remote build status.
               */
              pollInterval: 5
            )

            echo "Remote job has been triggered."
            echo "Initial remote build status: ${remoteBuildHandle.getBuildStatus()}"

            /*
             * Wait until the remote build leaves the queue and gets a concrete build URL.
             */
            timeout(time: 30, unit: 'MINUTES') {
              waitUntil {
                try {
                  remoteBuildHandle.updateBuildStatus()
                } catch (Throwable e) {
                  echo "Waiting for remote build URL. Status update warning: ${e.getMessage()}"
                }

                remoteBuildUrl = remoteBuildHandle.getBuildUrl()?.toString()

                if (remoteBuildUrl?.trim()) {
                  return true
                }

                echo "Remote build has not started yet. Current status: ${remoteBuildHandle.getBuildStatus()}"
                sleep time: 2, unit: 'SECONDS'
                return false
              }
            }

            remoteBuildUrl = remoteBuildUrl.trim()
            echo "Remote build URL: ${remoteBuildUrl}"

            /*
             * Stream the remote Jenkins console log into this pipeline console.
             *
             * This uses:
             *
             *   /logText/progressiveText?start=<offset>
             *
             * Jenkins returns:
             *
             *   X-Text-Size: <next offset>
             *   X-More-Data: true
             *
             * The loop keeps reading from the next offset until X-More-Data is no longer true.
             */
            withEnv([
              "REMOTE_BUILD_URL=${remoteBuildUrl}",
              "LOG_POLL_SECONDS=${logPollSeconds}"
            ]) {
              sh label: 'Stream remote Jenkins console log', script: '''#!/usr/bin/env bash
set -euo pipefail
set +x

: "${REMOTE_BUILD_URL:?REMOTE_BUILD_URL is required}"
: "${REMOTE_JENKINS_USER:?REMOTE_JENKINS_USER is required}"
: "${REMOTE_JENKINS_TOKEN:?REMOTE_JENKINS_TOKEN is required}"

offset=0
poll_seconds="${LOG_POLL_SECONDS:-2}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

headers="${tmp_dir}/headers"
body="${tmp_dir}/body"

echo "========== Remote Jenkins console log start =========="

while true; do
  : > "$headers"
  : > "$body"

  http_code="$(
    curl -sS -L \
      --connect-timeout 10 \
      --max-time 120 \
      -u "$REMOTE_JENKINS_USER:$REMOTE_JENKINS_TOKEN" \
      -D "$headers" \
      -o "$body" \
      -w "%{http_code}" \
      "${REMOTE_BUILD_URL%/}/logText/progressiveText?start=${offset}"
  )" || {
    rc=$?
    echo "ERROR: curl failed while reading the remote console log. curl exit code: ${rc}"
    exit "$rc"
  }

  case "$http_code" in
    2*)
      ;;
    *)
      echo "ERROR: failed to read the remote console log. HTTP status code: ${http_code}"
      echo "--- Response headers ---"
      sed -n '1,80p' "$headers" || true
      echo "--- Response body preview ---"
      sed -n '1,120p' "$body" || true
      exit 1
      ;;
  esac

  if [ -s "$body" ]; then
    cat "$body"
  fi

  next_offset="$(
    tr -d '\\r' < "$headers" \
      | awk 'tolower($1)=="x-text-size:" {print $2}' \
      | tail -n 1
  )"

  more_data="$(
    tr -d '\\r' < "$headers" \
      | awk 'tolower($1)=="x-more-data:" {print tolower($2)}' \
      | tail -n 1
  )"

  if [ -z "$next_offset" ]; then
    echo
    echo "ERROR: Jenkins did not return the X-Text-Size header."
    echo "The response may not be Jenkins progressive text output."
    echo "--- Response headers ---"
    sed -n '1,80p' "$headers" || true
    echo "--- Response body preview ---"
    sed -n '1,120p' "$body" || true
    exit 1
  fi

  offset="$next_offset"

  if [ "$more_data" = "true" ]; then
    sleep "$poll_seconds"
  else
    break
  fi
done

echo
echo "========== Remote Jenkins console log end =========="
'''
            }

            /*
             * Do not use curl to query /api/xml or /api/json for the final result.
             * The previous failure happened because that final curl request returned HTTP 403.
             *
             * Use the triggerRemoteJob handle instead.
             */
            String remoteBuildStatus = null
            String remoteBuildResult = null

            timeout(time: 5, unit: 'MINUTES') {
              waitUntil {
                try {
                  remoteBuildHandle.updateBuildStatus()
                } catch (Throwable e) {
                  echo "Final remote status update warning: ${e.getMessage()}"
                }

                remoteBuildStatus = remoteBuildHandle.getBuildStatus()?.toString()
                remoteBuildResult = remoteBuildHandle.getBuildResult()?.toString()

                echo "Remote build status: ${remoteBuildStatus}, result: ${remoteBuildResult}"

                if (remoteBuildResult?.trim()) {
                  return true
                }

                sleep time: 2, unit: 'SECONDS'
                return false
              }
            }

            switch (remoteBuildResult) {
              case 'SUCCESS':
                echo "Remote build finished successfully."
                break

              case 'UNSTABLE':
                echo "Remote build finished with UNSTABLE. Marking this pipeline as UNSTABLE."
                currentBuild.result = 'UNSTABLE'
                break

              case 'FAILURE':
              case 'ABORTED':
              case 'NOT_BUILT':
                error "Remote build finished with result: ${remoteBuildResult}"

              default:
                error "Remote build finished with an unknown result. Status: ${remoteBuildStatus}, result: ${remoteBuildResult}"
            }
          }
        }
      }
    }
  }
}
