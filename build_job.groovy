def triggerAndStreamRemoteJob(Map config) {
  String remoteJenkinsName = config.remoteJenkinsName
  String remoteCredentialId = config.remoteCredentialId
  String remoteJobName = config.remoteJobName
  String displayName = config.displayName ?: remoteJobName

  Map<String, Object> remoteParameters = config.remoteParameters ?: [:]

  int logPollSeconds = (config.logPollSeconds ?: 2) as int
  int statusPollSeconds = (config.statusPollSeconds ?: 5) as int

  String remoteLogSourceEncoding = config.remoteLogSourceEncoding ?: 'UTF-8'

  boolean abortTriggeredJob = config.containsKey('abortTriggeredJob')
    ? config.abortTriggeredJob as boolean
    : true

  boolean allowUnstable = config.containsKey('allowUnstable')
    ? config.allowUnstable as boolean
    : false

  def remoteBuildHandle = null
  String remoteBuildUrl = null

  withCredentials([
    usernamePassword(
      credentialsId: remoteCredentialId,
      usernameVariable: 'REMOTE_JENKINS_USER',
      passwordVariable: 'REMOTE_JENKINS_TOKEN'
    )
  ]) {
    Map triggerArgs = [
      remoteJenkinsName: remoteJenkinsName,
      job: remoteJobName,
      auth: [
        $class: 'CredentialsAuth',
        credentials: remoteCredentialId
      ],
      blockBuildUntilComplete: false,
      enhancedLogging: false,
      shouldNotFailBuild: true,
      abortTriggeredJob: abortTriggeredJob,
      pollInterval: statusPollSeconds
    ]

    if (remoteParameters != null && !remoteParameters.isEmpty()) {
      triggerArgs.parameters = remoteParameters
    }

    remoteBuildHandle = triggerRemoteJob(triggerArgs)

    echo "${displayName} has been triggered."
    echo "${displayName} initial remote status: ${remoteBuildHandle.getBuildStatus()}"

    timeout(time: 30, unit: 'MINUTES') {
      waitUntil {
        try {
          remoteBuildHandle.updateBuildStatus()
        } catch (Exception e) {
          echo "Waiting for ${displayName} build URL. Status update warning: ${e.getMessage()}"
        }

        remoteBuildUrl = remoteBuildHandle.getBuildUrl()?.toString()

        if (remoteBuildUrl?.trim()) {
          return true
        }

        echo "${displayName} has not started yet. Current status: ${remoteBuildHandle.getBuildStatus()}"
        sleep time: 2, unit: 'SECONDS'
        return false
      }
    }

    remoteBuildUrl = remoteBuildUrl.trim()
    echo "${displayName} build URL: ${remoteBuildUrl}"

    withEnv([
      "REMOTE_BUILD_URL=${remoteBuildUrl}",
      "LOG_POLL_SECONDS=${logPollSeconds}",
      "REMOTE_LOG_SOURCE_ENCODING=${remoteLogSourceEncoding}",
      "LANG=C.UTF-8",
      "LC_ALL=C.UTF-8"
    ]) {
      sh label: "Stream ${displayName} console log", encoding: 'UTF-8', script: '''#!/usr/bin/env bash
set -euo pipefail
set +x

: "${REMOTE_BUILD_URL:?REMOTE_BUILD_URL is required}"
: "${REMOTE_JENKINS_USER:?REMOTE_JENKINS_USER is required}"
: "${REMOTE_JENKINS_TOKEN:?REMOTE_JENKINS_TOKEN is required}"

offset=0
poll_seconds="${LOG_POLL_SECONDS:-2}"
source_encoding="${REMOTE_LOG_SOURCE_ENCODING:-UTF-8}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

headers="${tmp_dir}/headers"
body="${tmp_dir}/body"
converted_body="${tmp_dir}/body.utf8"

print_log_body() {
  if [ ! -s "$body" ]; then
    return 0
  fi

  if [ "$source_encoding" = "UTF-8" ] || [ "$source_encoding" = "utf-8" ]; then
    cat "$body"
    return 0
  fi

  if command -v iconv >/dev/null 2>&1; then
    if iconv -f "$source_encoding" -t UTF-8 "$body" > "$converted_body" 2>/dev/null; then
      cat "$converted_body"
      return 0
    fi
  fi

  cat "$body"
}

echo "========== Remote Jenkins console log start =========="

while true; do
  : > "$headers"
  : > "$body"
  : > "$converted_body"

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

  print_log_body

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

    String remoteBuildStatus = null
    String remoteBuildResult = null

    timeout(time: 5, unit: 'MINUTES') {
      waitUntil {
        try {
          remoteBuildHandle.updateBuildStatus()
        } catch (Exception e) {
          echo "Final ${displayName} status update warning: ${e.getMessage()}"
        }

        remoteBuildStatus = remoteBuildHandle.getBuildStatus()?.toString()
        remoteBuildResult = remoteBuildHandle.getBuildResult()?.toString()

        echo "${displayName} remote status: ${remoteBuildStatus}, result: ${remoteBuildResult}"

        if (remoteBuildResult?.trim()) {
          return true
        }

        sleep time: 2, unit: 'SECONDS'
        return false
      }
    }

    switch (remoteBuildResult) {
      case 'SUCCESS':
        echo "${displayName} finished successfully."
        break

      case 'UNSTABLE':
        if (allowUnstable) {
          echo "${displayName} finished with UNSTABLE. Marking this pipeline as UNSTABLE."
          currentBuild.result = 'UNSTABLE'
          break
        }
        error "${displayName} finished with UNSTABLE."

      case 'FAILURE':
      case 'ABORTED':
      case 'NOT_BUILT':
        error "${displayName} finished with result: ${remoteBuildResult}"

      default:
        error "${displayName} finished with an unknown result. Status: ${remoteBuildStatus}, result: ${remoteBuildResult}"
    }

    return [
      jobName: remoteJobName,
      buildUrl: remoteBuildUrl,
      status: remoteBuildStatus,
      result: remoteBuildResult
    ]
  }
}

def buildRun = null
def deployRun = null

pipeline {
  agent any

  options {
    timestamps()
  }

  stages {
    stage('Step 1 - Trigger Build Pipeline') {
      steps {
        script {
          String remoteJenkinsName = 'remoteJenkins'
          String remoteCredentialId = 'remote-jenkins-api-token'

          /*
           * Minimal encoding fix:
           * Use UTF-8 by default.
           * If the remote Jenkins job log is produced by a Windows GBK/GB18030 environment,
           * change this value to 'GB18030'.
           */
          String remoteLogSourceEncoding = 'UTF-8'

          buildRun = triggerAndStreamRemoteJob(
            remoteJenkinsName: remoteJenkinsName,
            remoteCredentialId: remoteCredentialId,
            remoteJobName: 'folder/build-pipeline',
            displayName: 'Remote Build Pipeline',
            remoteLogSourceEncoding: remoteLogSourceEncoding,
            logPollSeconds: 2,
            statusPollSeconds: 5,
            allowUnstable: false,
            abortTriggeredJob: true,
            remoteParameters: [
              ENV: 'dev',
              BRANCH: 'main'
            ]
          )

          echo "Remote build pipeline URL: ${buildRun.buildUrl}"
        }
      }
    }

    stage('Step 2 - Trigger Deploy Pipeline') {
      steps {
        script {
          String remoteJenkinsName = 'remoteJenkins'
          String remoteCredentialId = 'remote-jenkins-api-token'

          /*
           * Keep this value the same as the build stage unless the deploy job uses a different log encoding.
           */
          String remoteLogSourceEncoding = 'UTF-8'

          deployRun = triggerAndStreamRemoteJob(
            remoteJenkinsName: remoteJenkinsName,
            remoteCredentialId: remoteCredentialId,
            remoteJobName: 'folder/deploy-pipeline',
            displayName: 'Remote Deploy Pipeline',
            remoteLogSourceEncoding: remoteLogSourceEncoding,
            logPollSeconds: 2,
            statusPollSeconds: 5,
            allowUnstable: false,
            abortTriggeredJob: true,
            remoteParameters: [
              ENV: 'dev',
              BUILD_PIPELINE_URL: buildRun.buildUrl
            ]
          )

          echo "Remote deploy pipeline URL: ${deployRun.buildUrl}"
        }
      }
    }
  }
}
